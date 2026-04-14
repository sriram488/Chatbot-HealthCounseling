import os

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from openai import OpenAI

from .dataset_reply import reply_from_dataset


def _openai_reply(user_input: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_input}],
    )
    return response.choices[0].message.content or ""


@require_http_methods(["GET", "POST"])
def chatbot_response(request):
    if request.method == "POST":
        user_input = (request.POST.get("user_input") or "").strip()
        if not user_input:
            return JsonResponse({"reply": "Please enter a message."}, status=400)

        chatbot_reply: str | None = None
        source = "openai"

        try:
            chatbot_reply = _openai_reply(user_input)
        except Exception:
            chatbot_reply = None

        if chatbot_reply is None:
            fallback = reply_from_dataset(user_input)
            if fallback is None:
                return JsonResponse(
                    {
                        "reply": (
                            "I’m not able to generate a reply right now. "
                            "Add OPENAI_API_KEY to your .env file, or add the counselling dataset file."
                        )
                    },
                    status=503,
                )
            chatbot_reply = fallback
            source = "dataset"

        payload = {"reply": chatbot_reply}
        if os.getenv("CHATBOT_DEBUG_SOURCE"):
            payload["source"] = source
        return JsonResponse(payload)

    return render(request, "chatbot/chat.html")
