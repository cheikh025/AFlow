import workspace.FullStack.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

INSTRUCTION = (
    "Solve the following coding problem. "
    "Wrap your complete solution in a markdown code fence with the appropriate language tag "
    "(e.g. ```cpp ... ``` for C++, ```python ... ``` for Python, ```java ... ``` for Java). "
    "Output only the code block, no extra explanation.\n\n"
)


class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(
            input=problem,
            instruction=INSTRUCTION,
        )
        return solution["response"], self.llm.get_usage_summary()["total_cost"]
