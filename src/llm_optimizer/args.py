import abc
import ast
import itertools
import re
import typing as t
from enum import Enum, auto

from pydantic import BaseModel, computed_field, model_validator

T = t.TypeVar("T")
ValueT = t.Union[str, float, int, bool]

TYPE_MAP = {
    "int": int,
    "str": str,
    "bool": bool,
    "float": float,
}


class ArgScope(Enum):
    CLIENT = auto()
    SERVER = auto()


def _normalize_arg_name(arg_name: str) -> str:
    if arg_name.startswith("--"):
        arg_name = arg_name[2:]
    return arg_name.replace("-", "_")


def _convert_name_to_arg_name(name: str) -> str:
    if not name.startswith("--"):
        name = "--" + name
    return name.replace("_", "-")


class BaseArg(BaseModel, abc.ABC):
    scope: ArgScope

    @abc.abstractmethod
    def generate_cmd_args(self) -> t.Generator[str, t.Any, None]:
        pass

    @abc.abstractmethod
    def generate_kv_pairs(self) -> t.Generator[tuple[str, t.Any], t.Any, None]:
        pass


class Arg(BaseArg, t.Generic[T]):
    """Represents a single argument with a specific value (e.g., max_model_len=4096)."""

    name: str
    value: T

    def generate_cmd_args(self) -> t.Generator[str, t.Any, None]:
        arg_name = _convert_name_to_arg_name(self.name)
        if self.arg_type is bool:
            if self.value:
                yield arg_name
            return

        yield "=".join([arg_name, str(self.value)])

    def generate_kv_pairs(self) -> t.Generator[tuple[str, T], t.Any, None]:
        k = _normalize_arg_name(self.name)
        yield (k, self.value)

    @computed_field
    @property
    def arg_type(self) -> type[T]:
        return self.__class__.__pydantic_generic_metadata__["args"][0]


class CompositeArg(BaseArg):
    """Represents a set of coupled arguments (e.g., tp_size=2 and dp_size=4)."""

    args: list[Arg]

    def generate_cmd_args(self) -> t.Generator[str, t.Any, None]:
        for arg in self.args:
            yield from arg.generate_cmd_args()

    def generate_kv_pairs(self) -> t.Generator[tuple[str, t.Any], t.Any, None]:
        for arg in self.args:
            yield from arg.generate_kv_pairs()


def get_all_cmd_args(args: list[BaseArg]) -> list[str]:
    return [s for arg in args for s in arg.generate_cmd_args()]


def get_all_kv_pairs(args: list[BaseArg]) -> list[tuple[str, t.Any]]:
    return [s for arg in args for s in arg.generate_kv_pairs()]


class ArgConfig(BaseModel, t.Generic[T]):
    """Pre-defines the type and allowed values for an argument."""

    name: str
    allowed_values: t.Optional[set[T]] = None

    @computed_field
    @property
    def arg_type(self) -> type[T]:
        return self.__class__.__pydantic_generic_metadata__["args"][0]


ConfigsDict = dict[str, ArgConfig]


class ArgSet(BaseModel):
    """
    Represents a parsed argument definition from the input string, including all its possible values.
    Example: a single 'max_model_len=[4096, 8192]' part from the input string.
    """

    scope: ArgScope
    name: t.Union[str, tuple[str, ...]]
    arg_type: t.Union[type[ValueT], tuple[type[ValueT], ...]]
    values: t.Union[list[ValueT], list[tuple[ValueT, ...]]]

    @model_validator(mode="after")
    def validate_values_against_type(self):
        """Ensures all provided values are compatible with the determined arg_type."""
        is_composite = isinstance(self.name, tuple)

        for i, value in enumerate(self.values):
            if is_composite:
                # For composite args, value should be a tuple/list
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"Value at index {i} for composite arg '{self.name}' must be a list/tuple, got {type(value)}"
                    )
                if len(value) != len(self.name):
                    raise ValueError(
                        f"Value tuple {value} has length {len(value)}, but composite arg '{self.name}' requires length {len(self.name)}"
                    )
                # Check each sub-value's type
                for j, sub_val in enumerate(value):
                    expected_type = self.arg_type[j]
                    if not isinstance(sub_val, expected_type):
                        raise TypeError(
                            f"Value '{sub_val}' for arg '{self.name[j]}' must be of type {expected_type.__name__}, but got {type(sub_val).__name__}"
                        )
            else:
                # For single args, check the value's type directly
                if not isinstance(value, self.arg_type):
                    raise TypeError(
                        f"Value '{value}' for arg '{self.name}' must be of type {self.arg_type.__name__}, but got {type(value).__name__}"
                    )
        return self

    def get_all_possible_arg_values(self) -> list[BaseArg]:
        """
        Generates a list of concrete Arg or CompositeArg instances, one for each possible value.
        """
        possible_args = []
        is_composite = isinstance(self.name, tuple)

        for value in self.values:
            if is_composite:
                # Create a CompositeArg containing multiple simple Args
                child_args = [
                    Arg[self.arg_type[i]](
                        scope=self.scope, name=self.name[i], value=sub_value
                    )
                    for i, sub_value in enumerate(value)
                ]
                possible_args.append(CompositeArg(scope=self.scope, args=child_args))
            else:
                # Create a simple Arg
                possible_args.append(
                    Arg[self.arg_type](scope=self.scope, name=self.name, value=value)
                )

        return possible_args

    def to_arg_str_list(self) -> list[str]:
        """
        Convert an ArgSet to a list of argument strings.

        For single-value composite arguments, returns flattened individual arguments.
        For all other cases, returns a single-element list with the composite format.

        Examples:
        - ArgSet(name="max_model_len", values=[4096, 8192], arg_type=int) -> ["max_model_len=[4096, 8192]"]
        - ArgSet(name=("tp_size", "dp_size"), values=[(1, 8)], arg_type=(int, int)) -> ["tp_size=1", "dp_size=8"]
        - ArgSet(name=("tp_size", "dp_size"), values=[(2,4),(4,2)], arg_type=(int, int)) -> ["tp_size*dp_size=[(2, 4), (4, 2)]"]

        Returns:
            list[str]: List of argument strings
        """
        is_composite = isinstance(self.name, tuple)

        # Special case: single-value composite args should be flattened
        if is_composite and len(self.values) == 1:
            value = self.values[0]
            if isinstance(value, (list, tuple)):
                # Return flattened individual arguments
                result = []
                for i, sub_value in enumerate(value):
                    result.append(f"{self.name[i]}={sub_value}")
                return result
            else:
                # Single non-tuple value for composite - shouldn't happen but handle gracefully
                return [f"{self.name[0]}={value}"]

        # For all other cases, use the original composite format
        return [self.to_arg_str()]

    def to_arg_str(self) -> str:
        """
        Convert an ArgSet back to an argument string format.

        This is the inverse operation of parse_arg_str.

        Examples:
        - ArgSet(name="max_model_len", values=[4096, 8192], arg_type=int) -> "max_model_len=[4096, 8192]"
        - ArgSet(name="enable_ep_moe", values=[True, False], arg_type=bool) -> "enable_ep_moe=[True, False]"
        - ArgSet(name=("tp_size", "dp_size"), values=[(2,4),(4,2),(8,1)], arg_type=(int, int)) -> "tp_size*dp_size=[(2, 4), (4, 2), (8, 1)]"

        Returns:
            str: Argument string in the format expected by parse_arg_str
        """
        is_composite = isinstance(self.name, tuple)

        # Build the key part
        if is_composite:
            key_part = "*".join(self.name)
        else:
            key_part = self.name

        # Build the value part
        if len(self.values) == 1:
            # Single value - no brackets needed for simple values
            value = self.values[0]
            if is_composite:
                # For single-value composite args, we could flatten them here too
                # but for now keep composite format to maintain parse/to_arg_str symmetry
                if isinstance(value, (list, tuple)):
                    value_str = str(tuple(value))
                else:
                    value_str = str(value)
            else:
                # For simple single values, don't add brackets
                value_str = str(value)
        else:
            # Multiple values - always use list format
            if is_composite:
                # For composite args, convert each value to tuple format
                formatted_values = []
                for value in self.values:
                    if isinstance(value, (list, tuple)):
                        formatted_values.append(tuple(value))
                    else:
                        formatted_values.append(value)
                value_str = str(formatted_values)
            else:
                # For simple args with multiple values
                value_str = str(self.values)

        return f"{key_part}={value_str}"


def arg_sets_to_arg_str(arg_sets: list['ArgSet']) -> str:
    """
    Convert a list of ArgSets to a semicolon-separated argument string.

    Uses flattened output for single-value composite arguments.

    Args:
        arg_sets: List of ArgSet objects

    Returns:
        str: Semicolon-separated argument string (e.g., "max_model_len=4096;tp_size=1;dp_size=8")
    """
    if not arg_sets:
        return ""

    # Collect all argument strings, flattening single-value composites
    all_arg_strs = []
    for arg_set in arg_sets:
        all_arg_strs.extend(arg_set.to_arg_str_list())

    return ";".join(all_arg_strs)


def parse_args_str(
    args_str: str,
    scope: ArgScope,
    configs: t.Optional[dict[str, ArgConfig]],
    strict: bool,
) -> list[ArgSet]:
    if not args_str:
        return []

    parts = re.split(";+", args_str)
    return [
        parse_arg_str(part, scope=scope, configs=configs, strict=strict)
        for part in parts
    ]


def parse_arg_str(
    arg_str: str,
    scope: ArgScope,
    configs: t.Optional[ConfigsDict],
    strict: bool,
) -> ArgSet:
    """
    - "max_model_len=[4096, 8192]" -> ArgSet(name="max_model_len", values=[4096, 8192], arg_type=int)
    - "--eanble-ep-moe:bool=[True, False]" -> ArgSet(name="enable_ep_moe", values=[True, False], arg_type=bool)
    - "tp_size*dp_size:(int, int)=[[2,4],[4,2],[8,1]]" -> ArgSet(name=("tp_size", "dp_size"), values=[[2, 4], [4, 2], [8, 1]], arg_type=(int, int))
    """

    if strict and not configs:
        raise ValueError("Strict mode require predefined configs")

    scope_name = "server" if scope == ArgScope.SERVER else "client"

    arg_str = arg_str.strip()
    if "=" not in arg_str:
        raise ValueError(
            f"Invalid argument string format: '{arg_str}'. Must contain '='."
        )

    key_part, raw_value = arg_str.split("=", 1)

    names, arg_types_from_annotation = _parse_key_and_type_annotation(key_part)
    is_composite = isinstance(names, tuple)

    # Check for unknown args in strict mode
    arg_keys_to_check = names if is_composite else (names,)
    if strict:
        for name in arg_keys_to_check:
            if name not in configs:
                raise ValueError(
                    f"{scope_name} argument '{name}' not found in provided configurations and running in strict mode."
                )

    values_list = []
    raw_value_stripped = raw_value.strip()

    if raw_value_stripped.startswith("range("):
        try:
            values_list = _parse_range_str(raw_value_stripped)
        except ValueError as e:
            raise ValueError(f"Failed to parse range for '{key_part}': {e}")
    else:
        try:
            loaded_val = ast.literal_eval(raw_value_stripped)
            values_list = loaded_val if isinstance(loaded_val, list) else [loaded_val]
        except (ValueError, SyntaxError):
            # Fallback for unquoted raw strings like 'model_path=Qwen/Qwen3...'
            values_list = [raw_value]

    # priority of type inference:
    # 1. from user provided type annotation (e.g. `max_model_len:int`)
    # 2. from predefined configurations
    # 3. infer from provided values
    arg_types = arg_types_from_annotation
    if arg_types is None and configs:
        try:
            if is_composite:
                config_types = tuple(configs[n].arg_type for n in names)
                if len(config_types) == len(names):
                    arg_types = config_types
            elif names in configs:
                arg_types = configs[names].arg_type
        except KeyError:
            pass  # we will infer the types later

    if strict and arg_types_from_annotation and configs:
        config_type = None
        try:
            if is_composite:
                if all(n in configs for n in names):
                    config_type = tuple(configs[n].arg_type for n in names)
            elif names in configs:
                config_type = configs[names].arg_type
            if config_type and config_type != arg_types_from_annotation:
                raise TypeError(
                    f"Type annotation '{arg_types_from_annotation}' for '{names}' conflicts with config type '{config_type}' in strict mode."
                )
        except KeyError:
            pass  # No config for this key, so no conflict

    # infer types if not decided
    if arg_types is None:
        if not values_list:
            raise ValueError(
                f"Cannot determine type for '{names}': no type annotation and no values provided."
            )
        first_val = values_list[0]
        if is_composite:
            if not isinstance(first_val, (list, tuple)):
                raise ValueError(
                    f"Cannot infer composite types for '{key_part}', first value is not a list/tuple."
                )
            inferred_types = tuple(type(v) for v in first_val)
            arg_types = tuple(
                int
                if t == float
                and all(
                    isinstance(v_tuple[i], float) and v_tuple[i] == int(v_tuple[i])
                    for v_tuple in values_list
                )
                else t
                for i, t in enumerate(inferred_types)
            )
        else:
            inferred_type = type(first_val)
            if inferred_type == float and all(
                isinstance(v, float) and v == int(v) for v in values_list
            ):
                arg_types = int
            else:
                arg_types = inferred_type

    coerced_values = []
    for val in values_list:
        if is_composite:
            coerced_values.append(
                tuple(_coerce_value(v, t) for v, t in zip(val, arg_types))
            )
        else:
            coerced_values.append(_coerce_value(val, arg_types))

    return ArgSet(scope=scope, name=names, arg_type=arg_types, values=coerced_values)


def _parse_range_str(range_str: str) -> list[int]:
    if not (range_str.startswith("range(") and range_str.endswith(")")):
        raise ValueError(f"Malformed range string: {range_str}")
    content = range_str[len("range(") : -1]
    parts = content.split(",")
    if not (2 <= len(parts) <= 3):
        raise ValueError(
            f"range() expects 2 or 3 arguments, but got {len(parts)} in '{range_str}'"
        )
    try:
        args = [int(p.strip()) for p in parts]
    except ValueError:
        raise ValueError(
            f"Invalid non-integer argument found in range string: '{range_str}'"
        )
    return list(range(*args))


def _parse_key_and_type_annotation(
    key_part: str,
) -> tuple[t.Union[str, tuple[str, ...]], t.Optional[t.Union[type, tuple[type, ...]]]]:
    """
    Parses the part of the string before the '=', e.g., 'tp_size*dp_size:(int, int)'.
    Returns a tuple of (names, types).
    """
    # Check for and separate the optional type annotation
    if ":" in key_part:
        name_spec, type_spec = key_part.rsplit(":", 1)
    else:
        name_spec, type_spec = key_part, None

    # Parse the name(s)
    is_composite = "*" in name_spec
    if is_composite:
        names = tuple(_normalize_arg_name(n) for n in name_spec.split("*"))
    else:
        names = _normalize_arg_name(name_spec)

    # Parse the type annotation if it exists
    arg_types = None
    if type_spec:
        type_spec = type_spec.strip()
        if is_composite:
            if not (type_spec.startswith("(") and type_spec.endswith(")")):
                raise ValueError(
                    f"Composite arg '{name_spec}' requires a tuple type like '(int, int)', got '{type_spec}'"
                )
            type_names = [t.strip() for t in type_spec[1:-1].split(",")]
            if len(type_names) != len(names):
                raise ValueError(
                    f"Type annotation count {len(type_names)} does not match name count {len(names)} for '{name_spec}'"
                )
            arg_types = tuple(TYPE_MAP[t_name] for t_name in type_names)
        else:
            if type_spec not in TYPE_MAP:
                raise ValueError(
                    f"Unknown type annotation '{type_spec}' for argument '{name_spec}'"
                )
            arg_types = TYPE_MAP[type_spec]

    return names, arg_types


def _coerce_value(val: t.Any, typ: type) -> t.Any:
    return typ(val)


def get_all_arg_combinations(
    client_args_sets: list[ArgSet],
    server_args_sets: list[ArgSet],
) -> list[tuple[BaseArg, ...]]:
    args_sets = server_args_sets + client_args_sets
    all_possible_arg_values = [
        argset.get_all_possible_arg_values() for argset in args_sets
    ]
    all_combinations = list(itertools.product(*all_possible_arg_values))
    return all_combinations
