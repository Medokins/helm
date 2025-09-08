from typing import Any, Dict, List
from threading import Lock
import os

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

from helm.common.cache import CacheConfig
from helm.tokenizers.caching_tokenizer import CachingTokenizer
from helm.common.hierarchical_logger import htrack_block
from helm.common.tokenization_request import TokenizationToken, TokenizationRequest

# TEMPORARY
from pathlib import Path
import pandas as pd, json, os

CSV_PATH = Path(os.getenv("WATSONX_TOKENIZER_LOG",
                          "/helm-evaluation/benchmark_output/runs/prod/watsonx_tokenizer_calls.csv"))
CSV_LOCK = Lock()


def _to_jsonable(obj):
    """Return something that json.dumps can handle."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _log_tokenizer_call(request, raw, result, tokens):
    row = {
        "request": json.dumps(_to_jsonable(request), ensure_ascii=False),
        "text": request.get("text", ""),
        "raw": json.dumps(_to_jsonable(raw), ensure_ascii=False),
        "result": json.dumps(_to_jsonable(result), ensure_ascii=False),
        "tokens": json.dumps(_to_jsonable(tokens), ensure_ascii=False),
    }
    with CSV_LOCK:
        pd.DataFrame([row]).to_csv(CSV_PATH, mode="a",
                                   header=not CSV_PATH.exists(), index=False)

# mistral tokenizer doesn't work for these
_SYMBOL_MAP = {
    "¥": "JPY ",
    "£": "GBP ",
    "€": "EUR ",
    "\u00A0": " ",
}


def _sanitize(text: str) -> str:
    for bad, good in _SYMBOL_MAP.items():
        text = text.replace(bad, good)
    return text
# END TEMPORARY


class WatsonxTokenizer(CachingTokenizer):
    _models = {}
    _lock = Lock()

    def __init__(
            self,
            cache_config: CacheConfig,
            tokenizer_name: str,
            api_key: str | None = None,
            project_id: str | None = None,
            url: str | None = None,
            model_id: str | None = None,
            **kwargs,
    ):
        super().__init__(cache_config)
        self._tokenizer_name = tokenizer_name
        self._model_id = model_id or tokenizer_name.split("/", 1)[1]
        self._api_key = api_key or os.getenv("WATSONX_API_KEY")
        self._project_id = project_id or os.getenv("WATSONX_PROJECT_DALLAS")
        self._url = url or "https://yp-qa.ml.cloud.ibm.com"

    def _get_model(self) -> ModelInference:
        with self._lock:
            if self._tokenizer_name not in self._models:
                creds = Credentials(api_key=self._api_key, url=self._url)
                with htrack_block(f"Loading watsonx model {self._tokenizer_name}"):
                    self._models[self._tokenizer_name] = ModelInference(
                        model_id=self._tokenizer_name,
                        credentials=creds,
                        project_id=self._project_id,
                    )
            return self._models[self._tokenizer_name]

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Low-level wrapper around watsonx-ai tokenization.

        If request["encode"] is True we need integer IDs, otherwise we need strings.
        Watsonx-ai returns IDs when return_tokens=False and strings when return_tokens=True.
        """
        want_ids = request["encode"]
        text = _sanitize(request["text"])
        model = self._get_model()

        try:
            raw = model.tokenize(
                prompt=text,
                return_tokens=not want_ids,
            )
            result = raw["result"][0] if isinstance(raw["result"], list) else raw["result"]
            tokens = result.get("token_ids") if want_ids else result.get("tokens")

            _log_tokenizer_call(request, raw, result, tokens)
            return {"tokens": tokens}

        except Exception as e:
            _log_tokenizer_call(request, str(e), None, None)
            raise

    def _tokenization_raw_response_to_tokens(
            self, response: Dict[str, Any], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        return [TokenizationToken(token) for token in response]

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        text = " ".join(map(str, request["tokens"]))
        return {"text": text}
