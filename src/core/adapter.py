from abc import ABC,abstractmethod


class TaskAdapterHandler(ABC):
    @abstractmethod
    def support(self,domain:str) -> bool:
        """support for rag"""
        return False

    @abstractmethod
    def few_shot_adapter(self,prompt) -> str:
        """can option to use professional prompt,ignore basic prompt"""
        return prompt

    def language_adapter(self,prompt) -> str:
        """language adapter for rag"""
        return prompt + "\n- **工作语言** 请使用用户输入的主要语言作为你的工作语言"

    def dispatch(self,prompt:str) -> str:
        """dispatch prompt"""
        return self.language_adapter(self.few_shot_adapter(prompt))


class CommonTaskAdapterHandler(TaskAdapterHandler):
    def support(self,domain:str) -> bool:
        return domain != "unenabled"

    def few_shot_adapter(self,prompt) -> str:
        return prompt

