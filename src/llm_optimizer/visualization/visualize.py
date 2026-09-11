#!/usr/bin/env python3
"""
Pareto LLM Optimizer Dashboard

This script starts a local HTTP server to serve an interactive visualization
page for benchmark data using Plotly.js.
"""

import argparse
import json
import logging
import typing as t
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from llm_optimizer.result_utils import normalize_result_record
from llm_optimizer.utils import InfinityToNullEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _common_metadata_value(records: list[dict], key: str):
    values = {record.get(key) for record in records}
    return values.pop() if len(values) == 1 else "mixed"


def _record_field_value(record: dict, field: str):
    config = record.get("config", {})
    for section in ("normalized_config", "server_args", "client_args"):
        value = config.get(section, {}).get(field)
        if value is not None:
            return value
    if field == "framework":
        return config.get("framework")
    return record.get("results", {}).get(field)


def convert_constraints_for_visualization(parsed_constraints):
    """
    Convert parsed SLO constraints to the format expected by visualize.py.

    Args:
        parsed_constraints: List of SLOConstraint objects

    Returns:
        List of constraint objects with operator information preserved
        Example: [{"op": "<", "value": 1000.0, "name": "mean_ttft_ms"}]
    """
    if not parsed_constraints:
        return []

    constraints_list = []

    for constraint in parsed_constraints:
        # Convert metric name to match benchmark results format
        metric_mapping = {
            "ttft": "ttft_ms",
            "itl": "itl_ms",
            "tpot": "tpot_ms",
            "e2e_latency": "e2e_latency_ms",
        }

        base_metric = metric_mapping.get(constraint.metric, constraint.metric)

        # Convert value to milliseconds if needed
        value_ms = constraint.value
        if constraint.unit == "s":
            value_ms = constraint.value * 1000

        # Create field name with stat_type prefix
        if constraint.stat_type == "mean":
            field_name = f"mean_{base_metric}"
        else:
            field_name = f"{constraint.stat_type}_{base_metric}"

        constraints_list.append(
            {"op": constraint.operator, "value": value_ms, "name": field_name}
        )

    return constraints_list


class ParetoLLMOptimizer:
    def __init__(self, config_file: str = "visualization_config.json"):
        self.config_file = config_file
        self.config = self.load_config(config_file)

    def load_config(self, config_file: str) -> dict[str, t.Any]:
        """Load visualization configuration from JSON file."""
        try:
            with open(config_file, encoding="utf-8") as f:
                config = t.cast(dict[str, t.Any], json.load(f))

            # Framework will be detected later in load_benchmark_data

            logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}

    def load_benchmark_data(self, data_file: str) -> dict:
        """Load benchmark data from JSON files."""
        data = []
        constraints: list[t.Any] = []
        metadata: dict[str, t.Any] = {}
        best_configurations: dict[str, t.Any] = {}
        json_files = [
            Path(name.strip()) for name in data_file.split(",") if name.strip()
        ]

        if not json_files:
            raise ValueError("At least one data file is required")

        metadata_by_file = []
        best_by_file = {}
        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    file_data = json.load(f)

                if not isinstance(file_data, dict) or "test_results" not in file_data:
                    raise ValueError(
                        "Invalid data format. Expected enhanced format with "
                        "'metadata', 'best_configurations', and 'test_results' fields."
                    )

                file_records = [
                    normalize_result_record(item) for item in file_data["test_results"]
                ]
                data.extend(file_records)
                file_metadata = file_data.get("metadata", {})
                metadata_by_file.append({"data_file": str(json_file), **file_metadata})
                file_best = file_data.get("best_configurations", {})
                if file_best:
                    best_by_file[str(json_file)] = file_best

                file_constraints = file_metadata.get("constraints", [])
                if not file_constraints:
                    for item in file_records:
                        if (
                            isinstance(item, dict)
                            and "constraints" in item
                            and item["constraints"]
                        ):
                            file_constraints = item["constraints"]
                            break
                if file_constraints and not constraints:
                    constraints = file_constraints
                elif file_constraints and file_constraints != constraints:
                    logger.warning(
                        "Input files contain different constraints; using the "
                        "first set."
                    )
                logger.info(f"Loaded {len(file_records)} records from {json_file}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                raise

        if len(metadata_by_file) == 1:
            metadata = dict(metadata_by_file[0])
            metadata.pop("data_file", None)
            best_configurations = next(iter(best_by_file.values()), {})
        else:
            metadata = {
                "total_tests": len(data),
                "runs": metadata_by_file,
                "model_tag": _common_metadata_value(metadata_by_file, "model_tag"),
                "gpu_type": _common_metadata_value(metadata_by_file, "gpu_type"),
                "gpu_count": _common_metadata_value(metadata_by_file, "gpu_count"),
            }
            best_configurations = best_by_file

        logger.info(
            f"Loaded {len(data)} benchmark records from {len(json_files)} file(s)"
        )
        logger.info(f"Found constraints: {constraints}")

        return {
            "data": data,
            "constraints": constraints,
            "data_files": json_files,
            "metadata": metadata,
            "best_configurations": best_configurations,
        }

    def get_field_options(
        self, data: t.Optional[list] = None
    ) -> dict[str, t.Any]:
        """Get field options by category, with disabled fields for missing data."""
        if not self.config or "fields" not in self.config:
            return {}

        categories = {}
        for field_id, field_info in self.config["fields"].items():
            if (
                data
                and field_info.get("optional")
                and not any(
                    _record_field_value(item, field_id) is not None for item in data
                )
            ):
                continue
            category = field_info.get("category", "other")
            if category not in categories:
                categories[category] = {
                    "label": self.config.get("categories", {})
                    .get(category, {})
                    .get("label", category.title()),
                    "description": self.config.get("categories", {})
                    .get(category, {})
                    .get("description", ""),
                    "fields": {},
                }

            categories[category]["fields"][field_id] = field_info

        return categories

    def get_field_categories(self) -> dict[str, t.Any]:
        """Get field category information for hover display."""
        if not self.config or "field_categories" not in self.config:
            return {}
        return t.cast(dict[str, t.Any], self.config["field_categories"])

    def create_html_page(self, data_dict: dict) -> str:
        """Create the HTML page with embedded data and configuration."""
        # Load the HTML template
        # Try to find template relative to this file's location
        current_dir = Path(__file__).parent
        template_path = current_dir / "template.html"
        if not template_path.exists():
            logger.error(f"Template file not found at {template_path}")
            return ""

        with open(template_path, encoding="utf-8") as f:
            html_content = f.read()

        # Prepare data for embedding
        data_json = json.dumps(data_dict["data"], indent=2, cls=InfinityToNullEncoder)
        constraints_json = json.dumps(
            data_dict["constraints"], indent=2, cls=InfinityToNullEncoder
        )
        config_json = json.dumps(self.config, indent=2, cls=InfinityToNullEncoder)
        field_options_json = json.dumps(
            self.get_field_options(data_dict["data"]),
            indent=2,
            cls=InfinityToNullEncoder,
        )
        field_categories_json = json.dumps(
            self.get_field_categories(), indent=2, cls=InfinityToNullEncoder
        )
        defaults_json = json.dumps(
            self.config.get("defaults", {}), indent=2, cls=InfinityToNullEncoder
        )
        metadata_json = json.dumps(
            data_dict.get("metadata", {}), indent=2, cls=InfinityToNullEncoder
        )
        best_configs_json = json.dumps(
            data_dict.get("best_configurations", {}),
            indent=2,
            cls=InfinityToNullEncoder,
        )

        # Get UI configuration
        ui_config = self.config.get("ui", {})

        # Replace placeholders in the template
        html_content = html_content.replace(
            "{title}", ui_config.get("title", "Pareto LLM Optimizer Dashboard")
        )
        html_content = html_content.replace(
            "{subtitle}",
            ui_config.get(
                "subtitle", "Interactive Performance Analysis & Optimization Tool"
            ),
        )
        html_content = html_content.replace(
            "{description}",
            ui_config.get(
                "description",
                "Select different metrics for X and Y axes to analyze performance "
                "relationships and identify Pareto optimal configurations. "
                "Hover over data points to see detailed configuration information.",
            ),
        )
        html_content = html_content.replace("{data_json}", data_json)
        html_content = html_content.replace("{constraints_json}", constraints_json)
        html_content = html_content.replace("{config_json}", config_json)
        html_content = html_content.replace("{field_options_json}", field_options_json)
        html_content = html_content.replace(
            "{field_categories_json}", field_categories_json
        )
        html_content = html_content.replace("{defaults_json}", defaults_json)
        html_content = html_content.replace("{metadata_json}", metadata_json)
        html_content = html_content.replace("{best_configs_json}", best_configs_json)

        return html_content

    def generate_dashboard(
        self, data_file: str, output_file: t.Optional[str] = None
    ) -> str:
        """Generate the dashboard HTML file."""
        # Load benchmark data
        data_dict = self.load_benchmark_data(data_file)

        # Create HTML page
        html_content = self.create_html_page(data_dict)

        # Write to file
        html_file = output_file if output_file else "pareto_llm_dashboard.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Created Pareto LLM Dashboard: {html_file}")
        return html_file

    def start_server(self, port: int = 8080):
        """Start the HTTP server."""
        try:
            server = HTTPServer(("localhost", port), SimpleHTTPRequestHandler)
            logger.info(f"Starting server at http://localhost:{port}")
            logger.info(
                f"Dashboard available at: http://localhost:{port}/pareto_llm_dashboard.html"
            )
            logger.info("Press Ctrl+C to stop the server")

            # Open the dashboard in the default browser
            import webbrowser

            webbrowser.open(f"http://localhost:{port}/pareto_llm_dashboard.html")

            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Error starting server: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pareto LLM Optimizer Dashboard")
    parser.add_argument(
        "--data", required=True, help="Data file path (e.g., sample_data.json)"
    )
    parser.add_argument(
        "--config", default="visualization_config.json", help="Configuration file path"
    )
    parser.add_argument("--port", type=int, default=8080, help="Server port")

    args = parser.parse_args()

    # Create optimizer instance
    optimizer = ParetoLLMOptimizer(args.config)

    # Generate dashboard
    optimizer.generate_dashboard(args.data)

    # Start server
    optimizer.start_server(args.port)


if __name__ == "__main__":
    main()
