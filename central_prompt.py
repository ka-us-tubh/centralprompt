from __future__ import annotations
from typing import Any, Dict, Optional, List, Literal
import os, re

try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None

try:
    from langfuse import get_client  # type: ignore
except Exception:
    get_client = None


class PromptProviderError(ValueError):
    pass


class PromptCreationError(Exception):
    pass


class PromptFetchError(Exception):
    pass


class PromptCompilationError(Exception):
    pass


__all__ = ["PromptHandle", "CentralPrompt"]


def _normalize_provider(provider: str) -> str:
    if not isinstance(provider, str):
        raise PromptProviderError("provider must be a string")
    value = provider.strip().lower()
    if value in ("mlflow", "ml-flow"):
        return "mlflow"
    if value in ("langfuse", "lang-fuse"):
        return "langfuse"
    raise PromptProviderError("Unsupported provider; use 'mlflow' or 'langfuse'")


class CentralPrompt:
    def __init__(
        self,
        provider: Literal["mlflow", "langfuse"],
        *,
        experiment: Optional[str] = None,
        load_env: bool = True,
    ) -> None:
        self.provider = _normalize_provider(provider)
        self.experiment = experiment
        self.load_env = load_env
        if self.experiment is not None and not isinstance(self.experiment, str):
            raise ValueError("experiment must be a string or None")
        self._langfuse_client = None
        if self.provider == "mlflow":
            # tracking_uri is intentionally sourced from environment (MLFLOW_TRACKING_URI)
            _ensure_mlflow(None, self.experiment, load_env=self.load_env)
        if self.provider == "langfuse":
            self._langfuse_client = _ensure_langfuse(self.load_env)

    def set_prompt(
        self,
        name: str,
        template: Any,
        *,
        prompt_type: Optional[str] = None,
        labels: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
        response_format: Optional[Any] = None,
        commit_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        _validate_name(name)
        _validate_template(template)
        _validate_labels(labels)
        if tags is not None and (not isinstance(tags, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in tags.items())):
            raise ValueError("tags must be a dict[str, str]")

        if self.provider == "mlflow":
            if not hasattr(mlflow, "genai"):
                raise ImportError("mlflow.genai is not available. Please upgrade mlflow to a version that supports genai.")
            kwargs: Dict[str, Any] = {
                "name": name,
                "template": template,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            if commit_message is not None:
                kwargs["commit_message"] = commit_message
            if tags is not None:
                kwargs["tags"] = tags
            try:
                prompt_obj = mlflow.genai.register_prompt(**kwargs)  # type: ignore[attr-defined]
            except Exception as e:
                raise PromptCreationError(f"MLflow prompt creation failed: {e}") from e
            return {"provider": self.provider, "name": getattr(prompt_obj, "name", name), "version": getattr(prompt_obj, "version", None)}

        if self.provider == "langfuse":
            derived_type = prompt_type or ("chat" if _is_chat_template(template) else "text")
            if derived_type not in ("text", "chat"):
                raise ValueError("prompt_type must be 'text' or 'chat'")
            if derived_type == "chat" and not _is_chat_template(template):
                raise ValueError("For prompt_type='chat', template must be a list of {role, content} dicts")
            if derived_type == "text" and not isinstance(template, str):
                raise ValueError("For prompt_type='text', template must be a string")

            client = self._langfuse_client or _ensure_langfuse(self.load_env)
            payload: Dict[str, Any] = {
                "name": name,
                "type": derived_type,
                "prompt": template,
            }
            if labels is not None:
                payload["labels"] = labels
            try:
                client.create_prompt(**payload)
            except Exception as e:
                raise PromptCreationError(
                    f"Langfuse prompt creation failed: {e}. Ensure LANGFUSE credentials are configured."
                ) from e
            return {"provider": self.provider, "name": name}

        raise PromptProviderError("Unsupported provider; use 'mlflow' or 'langfuse'")

    def get_prompt(
        self,
        name: Optional[str] = None,
        *,
        version: Optional[int] = None,
        path: Optional[str] = None,
        label: Optional[str] = None,
    ) -> PromptHandle:
        if self.provider == "mlflow":
            target: Optional[str] = path
            if target is None:
                if not name or version is None:
                    raise ValueError("For mlflow, provide either 'path' or both 'name' and 'version'")
                if not isinstance(version, int) or version < 1:
                    raise ValueError("version must be a positive integer")
                target = f"prompts:/{name}/{version}"
            if not isinstance(target, str) or not re.match(r"^prompts:/[^/]+/\d+$", target):
                raise ValueError("mlflow 'path' must be of the form 'prompts:/<name>/<version>'")
            try:
                prompt_obj = mlflow.genai.load_prompt(target)  # type: ignore[attr-defined]
            except Exception as e:
                raise PromptFetchError(f"MLflow load_prompt failed for '{target}': {e}") from e
            # Extract name/version from the target path for better repr/debugging
            m = re.match(r"^prompts:/([^/]+)/([0-9]+)$", target)
            name_extracted = m.group(1) if m else None
            ver_extracted = int(m.group(2)) if m else None
            return PromptHandle(self.provider, prompt_obj, name=name_extracted, version=ver_extracted)

        if self.provider == "langfuse":
            client = self._langfuse_client or _ensure_langfuse(self.load_env)
            if not name or not isinstance(name, str):
                raise ValueError("For langfuse, 'name' is required and must be a string")
            if label is not None and not isinstance(label, str):
                raise ValueError("For langfuse, 'label' must be a string if provided")
            if version is not None and label is not None:
                raise ValueError("For langfuse, provide either 'version' or 'label', not both")
            try:
                if label is not None:
                    prompt_obj = client.get_prompt(name, version=version, label=label)
                else:
                    prompt_obj = client.get_prompt(name, version=version)
            except Exception as e:
                raise PromptFetchError(f"Langfuse get_prompt failed for '{name}': {e}") from e
            if prompt_obj is None:
                raise PromptFetchError(f"Langfuse prompt '{name}' not found")
            return PromptHandle(self.provider, prompt_obj, name=name, version=version)

        raise PromptProviderError("Unsupported provider; use 'mlflow' or 'langfuse'")


def _is_chat_template(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    for item in obj:
        if not isinstance(item, dict):
            return False
        if "role" not in item or "content" not in item:
            return False
        if not isinstance(item["role"], str) or not isinstance(item["content"], str):
            return False
    return True


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")


def _validate_template(template: Any) -> None:
    if isinstance(template, str):
        return
    if _is_chat_template(template):
        return
    raise ValueError("template must be a string or a chat template list of {role, content} dicts")


def _validate_labels(labels: Optional[List[str]]) -> None:
    if labels is None:
        return
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("labels must be a list of strings")


class PromptHandle:
    def __init__(
        self,
        provider: Literal["mlflow", "langfuse"],
        underlying: Any,
        *,
        name: Optional[str] = None,
        version: Optional[int] = None,
    ):
        self.provider = _normalize_provider(provider)
        self.underlying = underlying
        # Prefer explicit name/version if provided; otherwise try to read from underlying
        self.name = name if name is not None else getattr(self.underlying, "name", None)
        self.version = version if version is not None else getattr(self.underlying, "version", None)

    def __repr__(self) -> str:
        parts = [f"provider='{self.provider}'"]
        if self.name is not None:
            parts.append(f"name='{self.name}'")
        if self.version is not None:
            parts.append(f"version={self.version}")
        parts.append(f"underlying={type(self.underlying).__name__}")
        return f"PromptHandle({', '.join(parts)})"

    def compile(self, **variables: Any) -> Any:
        try:
            if self.provider == "mlflow":
                return self.underlying.format(**variables)
            if self.provider == "langfuse":
                return self.underlying.compile(**variables)
        except Exception as e:
            raise PromptCompilationError(f"Failed to compile prompt for provider '{self.provider}': {e}") from e
        raise PromptProviderError("Unsupported provider; use 'mlflow' or 'langfuse'")


def _ensure_mlflow(tracking_uri: Optional[str], experiment: Optional[str], *, load_env: bool = False) -> None:
    if mlflow is None:
        raise ImportError("mlflow is not installed")
    if load_env:
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    exp = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME")
    if exp:
        mlflow.set_experiment(exp)


def _ensure_langfuse(load_env: bool = True):
    if get_client is None:
        raise ImportError("langfuse is not installed")
    if load_env:
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
    return get_client()





