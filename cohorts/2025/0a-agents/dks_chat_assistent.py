
import json
from typing import Any
import ollama
import openai
from IPython.display import display, HTML
import markdown


class LLM:
    def __init__(self, llm_model: str):
        self.llm_model = llm_model

    def get_client(self):
        raise NotImplementedError("Subclasses must implement this method")

    def extract_response(self, llm_response: Any):
        raise NotImplementedError("Subclasses must implement this method")

    def get_llm_type(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_llm_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def decide_function_call(self, llm_response_entity: Any):
        raise NotImplementedError("Subclasses must implement this method")

    def extract_function_call(self, llm_response_entity: Any):
        raise NotImplementedError("Subclasses must implement this method")

    def decide_message(self, llm_response_entity: Any):
        raise NotImplementedError("Subclasses must implement this method")

    def extract_message(self, llm_response_entity: Any):
        raise NotImplementedError("Subclasses must implement this method")

    def function_call_output(self, tool_call_response: Any, result: Any):
        raise NotImplementedError("Subclasses must implement this method")


class OllamaLLM(LLM):
    def __init__(self, llm_model: str):
        super().__init__(llm_model)

    def get_client(self):
        return ollama.Client()

    def extract_response(self, llm_response: Any):
        return [llm_response.message]

    def get_llm_type(self):
        return "ollama"

    def get_llm_model(self):
        return self.llm_model

    def decide_function_call(self, llm_response_entity: Any):
        return llm_response_entity.tool_calls is not None

    def extract_function_call(self, llm_response_entity: Any):
        return llm_response_entity.tool_calls[0].function

    def decide_message(self, llm_response_entity: Any):
        return llm_response_entity.content != ""

    def extract_message(self, llm_response_entity: Any):
        return llm_response_entity.content

    def function_call_output(self, tool_call_response: Any, result):
        return {
            "role": "tool",
            "content": json.dumps(result, indent=2),
            "tool_name": tool_call_response.name,
        }


class GPTLLM(LLM):
    def __init__(self, llm_model: str):
        super().__init__(llm_model)

    def get_client(self):
        return openai.OpenAI()

    def extract_response(self, llm_response: Any):
        return llm_response.output

    def get_llm_type(self):
        return "gpt"

    def get_llm_model(self):
        return self.llm_model

    def decide_function_call(self, llm_response_entity: Any):
        return llm_response_entity.type == "function_call"

    def extract_function_call(self, llm_response_entity: Any):
        return llm_response_entity

    def decide_message(self, llm_response_entity: Any):
        return llm_response_entity.type == "message"

    def extract_message(self, llm_response_entity: Any):
        return llm_response_entity.content[0].text

    def function_call_output(self, tool_call_response: Any, result: Any):
        return {
            "type": "function_call_output",
            "call_id": tool_call_response.call_id,
            "output": json.dumps(result, indent=2),
        }


class Tools:
    def __init__(self):
        self.tools = {}
        self.functions = {}

    def add_tool(self, function, description):
        self.tools[function.__name__] = description
        self.functions[function.__name__] = function

    def get_tools(self, llm_type: str = "gpt"):
        if llm_type == "gpt":
            return list(self.tools.values())
        elif llm_type == "ollama":
            return list(self.functions.values())
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def function_call(self, tool_call_response, llm_object: LLM):
        function_name = tool_call_response.name
        if isinstance(tool_call_response.arguments, dict):
            arguments = tool_call_response.arguments
        else:
            arguments = json.loads(tool_call_response.arguments)

        f = self.functions[function_name]
        result = f(**arguments)

        return llm_object.function_call_output(tool_call_response, result)


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


class ChatInterface:
    def input(self):
        question = input("You:")
        return question

    def display(self, message):
        print(message)

    def display_function_call(self, entry, result):
        call_html = f"""
            <details>
            <summary>Function call:
            <tt>{entry.name}({shorten(entry.arguments)})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{entry}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result}</pre>
            </div>

            </details>
        """
        display(HTML(call_html))

    def display_response(self, entry):
        response_html = markdown.markdown(entry)
        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{response_html}</div>
            </div>
        """
        display(HTML(html))


class ChatAssistant:
    def __init__(self, tools, developer_prompt, chat_interface, client,
                 llm: LLM):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.client = client
        self.llm = llm

    def gpt(self, chat_messages):
        return self.client.responses.create(
            model=self.llm.get_llm_model(),
            input=chat_messages,
            tools=self.tools.get_tools(),
        )
    
    def ollama(self, chat_messages):
        return self.client.chat(
            model=self.llm.get_llm_model(),
            messages=chat_messages,
            tools=self.tools.get_tools(llm_type="ollama"),
            think=False,
        )

    def get_llm_response(self, chat_messages: list[dict]):
        if self.llm.get_llm_type() == "gpt":
            return self.gpt(chat_messages)
        elif self.llm.get_llm_type() == "ollama":
            return self.ollama(chat_messages)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm.get_llm_type()}")

    def run(self):
        chat_messages = [
            {"role": "developer", "content": self.developer_prompt},
        ]

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.strip().lower() == 'stop':
                self.chat_interface.display("Chat ended.")
                break

            message = {"role": "user", "content": question}
            chat_messages.append(message)

            while True:  # inner request loop
                response = self.get_llm_response(chat_messages)

                has_messages = False

                for entry in self.llm.extract_response(response):
                    chat_messages.append(entry)

                    if self.llm.decide_function_call(entry):
                        function_call_entity = self.llm.extract_function_call(entry)
                        result = self.tools.function_call(function_call_entity)
                        chat_messages.append(result)
                        self.chat_interface.display_function_call(
                            function_call_entity, result
                        )

                    elif self.llm.decide_message(entry):
                        message_entity = self.llm.extract_message(entry)
                        self.chat_interface.display_response(message_entity)
                        has_messages = True

                if has_messages:
                    break
