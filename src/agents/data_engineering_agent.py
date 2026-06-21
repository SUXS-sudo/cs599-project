from __future__ import annotations

from pathlib import Path

from src.services.heterogeneous_recipe_pipeline import run_heterogeneous_recipe_pipeline


class RecipeParsingAgent:
    """Offline Agent for heterogeneous recipe parsing, cleaning and graph construction."""

    def __init__(self, llm_client=None, enable_llm: bool = True) -> None:
        if enable_llm and llm_client is None:
            from src.services.llm_client import LLMClient

            llm_client = LLMClient()
        self.llm_client = llm_client
        self.enable_llm = enable_llm

    def run_pipeline(
        self,
        sources: list[Path],
        output_dir: Path,
        manual_minutes_per_record: float = 2.0,
        build_faiss: bool = True,
        import_mysql: bool = True,
        reset_mysql: bool = False,
        import_neo4j: bool = True,
        reset_neo4j: bool = False,
    ):
        return run_heterogeneous_recipe_pipeline(
            sources=sources,
            output_dir=output_dir,
            llm_client=self.llm_client,
            enable_llm=self.enable_llm,
            manual_minutes_per_record=manual_minutes_per_record,
            build_faiss=build_faiss,
            import_mysql=import_mysql,
            reset_mysql=reset_mysql,
            import_neo4j=import_neo4j,
            reset_neo4j=reset_neo4j,
        )
