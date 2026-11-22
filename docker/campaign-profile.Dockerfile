# Build with a context containing only administrator-reviewed campaign JSON files.
ARG CAMPAIGN_IMAGE
FROM ${CAMPAIGN_IMAGE}
COPY . /opt/llm-optimizer/campaign-profiles/
ENV LLM_OPTIMIZER_CAMPAIGN_PROFILES=/opt/llm-optimizer/campaign-profiles
