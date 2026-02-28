const API_BASE = window.location.origin;

const modal = document.getElementById("disclaimerModal");
const acceptBtn = document.getElementById("acceptBtn");
const messagesEl = document.getElementById("messages");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const exampleChips = document.querySelectorAll(".example-chip");

acceptBtn.addEventListener("click", () => {
    modal.classList.add("hidden");
    userInput.focus();
});

userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
});

exampleChips.forEach(chip => {
    chip.addEventListener("click", () => {
        userInput.value = chip.dataset.query;
        userInput.dispatchEvent(new Event("input"));
        sendMessage();
    });
});

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener("click", sendMessage);

clearBtn.addEventListener("click", () => {
    messagesEl.innerHTML = `
    <div class="welcome-block">
      <h1 class="welcome-title">How are you feeling?</h1>
      <p class="welcome-sub">Describe your symptoms in plain language. Pulse AI will search relevant medical literature and provide an informed response.</p>
    </div>`;
});

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || sendBtn.disabled) return;

    const welcome = messagesEl.querySelector(".welcome-block");
    if (welcome) welcome.remove();

    appendMessage("user", text);

    userInput.value = "";
    userInput.style.height = "auto";

    sendBtn.disabled = true;

    const typingEl = appendTyping();

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });

        typingEl.remove();

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Unknown error" }));
            appendMessage("ai", err.detail || "Something went wrong. Please try again.", true);
        } else {
            const data = await res.json();
            appendMessage("ai", data.response);
        }
    } catch (err) {
        typingEl.remove();
        appendMessage("ai", "Could not reach the server. Please check your connection and try again.", true);
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

function appendMessage(role, text, isError = false) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role === "user" ? "You" : "Pulse AI";

    const bubble = document.createElement("div");
    bubble.className = `bubble${isError ? " error-bubble" : ""}`;
    bubble.textContent = text;

    wrapper.appendChild(label);
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
}

function appendTyping() {
    const wrapper = document.createElement("div");
    wrapper.className = "message ai";

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "Pulse AI";

    const bubble = document.createElement("div");
    bubble.className = "typing-bubble";
    bubble.innerHTML = `
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  `;

    wrapper.appendChild(label);
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}
