from __future__ import annotations

from pathlib import Path


def test_chat_ui_has_session_scoped_conversation_history() -> None:
    root = Path(__file__).resolve().parent.parent
    html = (root / "app" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="conversationList"' in html
    assert 'id="newChat"' in html
    assert 'id="activeConversationTitle"' in html
    assert 'smartrecipe.conversations.v1' in html
    assert 'sessionId: `chat-${id}`' in html
    assert 'session_id: sessionInput.value.trim()' in html
    assert 'form.append("session_id", sessionInput.value.trim())' in html
    assert 'moreButton.textContent = "⋯"' in html
    assert 'removeButton.textContent = "删除"' in html
    assert 'moreButton.setAttribute("aria-expanded", "false")' in html
    assert 'closeConversationMenus()' in html
    assert 'addEventListener("contextmenu"' not in html
    assert 'deleteConversation(item.id)' in html
    assert html.count('href="/ui/database.html"') == 1
    assert 'value="demo-user"' not in html


def test_chat_ui_selects_vision_path_from_attached_image() -> None:
    root = Path(__file__).resolve().parent.parent
    html = (root / "app" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="attachImage"' in html
    assert 'id="imageFile" type="file" accept="image/*"' in html
    assert 'const selectedImage = imageFile.files[0] || null' in html
    assert 'selectedImage ? await submitImage(text, selectedImage) : await submitText(text)' in html
    assert 'form.append("image", file)' in html
    assert 'id="textMode"' not in html
    assert 'id="imageMode"' not in html
