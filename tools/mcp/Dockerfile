FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub

WORKDIR /app

COPY tools/mcp/requirements.txt /tmp/requirements.txt
COPY training/requirements-phase2.txt /tmp/requirements-phase2.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir torch && pip install --no-cache-dir -r /tmp/requirements-phase2.txt

COPY reference_vm.py /app/reference_vm.py
COPY dataset_pipeline.py /app/dataset_pipeline.py
COPY baseline_trainer.py /app/baseline_trainer.py
COPY curriculum_gate.py /app/curriculum_gate.py
COPY sft_export.py /app/sft_export.py
COPY vmbench_cli.py /app/vmbench_cli.py
COPY vmbench_product_surface.py /app/vmbench_product_surface.py
COPY branch_ranker.py /app/branch_ranker.py
COPY candidate_generator.py /app/candidate_generator.py
COPY search_runner.py /app/search_runner.py
COPY learned_branch_ranker.py /app/learned_branch_ranker.py
COPY vm_transition_verifier.py /app/vm_transition_verifier.py
COPY vmbench_demo_runtime.py /app/vmbench_demo_runtime.py
COPY tools/mcp/vmbench_mcp_server.py /app/tools/mcp/vmbench_mcp_server.py
COPY tools/mcp/vmbench_mcp_spec.md /app/tools/mcp/vmbench_mcp_spec.md
COPY README.md /app/README.md
COPY REPO_PRODUCT_MAP.md /app/REPO_PRODUCT_MAP.md
COPY PRODUCTIZATION_PLAN.md /app/PRODUCTIZATION_PLAN.md

WORKDIR /workspace

ENTRYPOINT ["python", "/app/tools/mcp/vmbench_mcp_server.py"]
