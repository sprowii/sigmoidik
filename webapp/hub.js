const state = {
    user: null,
    scope: "all",
    limit: 12,
    offset: 0,
    loading: false,
    reachedEnd: false,
    generating: false,
    currentTweakGameId: null,
};

const PROGRESS_PHRASES = [
    "🧠 Продумываем механику уровня...",
    "🎨 Добавляем спрайты и фон...",
    "🎵 Подключаем эффекты и звук...",
    "🕹️ Настраиваем управление и физику...",
    "🧪 Проверяем синтаксис и безопасность...",
];

let progressIntervalId = null;
let progressStep = 0;

const decodeCodes = (codes) => String.fromCharCode(...codes);

const routes = {
    games: decodeCodes([47, 97, 112, 105, 47, 103, 97, 109, 101, 115]),
    authSession: decodeCodes([47, 97, 112, 105, 47, 97, 117, 116, 104, 47, 115, 101, 115, 115, 105, 111, 110]),
    authLogin: decodeCodes([47, 97, 112, 105, 47, 97, 117, 116, 104, 47, 108, 111, 103, 105, 110]),
    authLogout: decodeCodes([47, 97, 112, 105, 47, 97, 117, 116, 104, 47, 108, 111, 103, 111, 117, 116]),
    tweakSuffix: decodeCodes([47, 116, 119, 101, 97, 107]),
};

const elements = {
    filters: document.getElementById("filters"),
    chips: Array.from(document.querySelectorAll(".filters .chip")),
    gameList: document.getElementById("game-list"),
    loadMore: document.getElementById("load-more"),
    sessionControls: document.getElementById("session-controls"),
    generatorForm: document.getElementById("generator-form"),
    generatorTextarea: document.getElementById("generator-idea"),
    generatorStatus: document.getElementById("generator-status"),
    generatorSubmit: document.getElementById("generator-submit"),
    modal: document.getElementById("login-modal"),
    modalForm: document.getElementById("login-form"),
    modalCancel: document.getElementById("login-cancel"),
    modalError: document.getElementById("login-error"),
    modalInput: document.getElementById("login-code"),
    tweakModal: document.getElementById("tweak-modal"),
    tweakForm: document.getElementById("tweak-form"),
    tweakCancel: document.getElementById("tweak-cancel"),
    tweakError: document.getElementById("tweak-error"),
    tweakTextarea: document.getElementById("tweak-instructions"),
    tweakTitle: document.getElementById("tweak-title"),
    gameTemplate: document.getElementById("game-card-template"),
};

async function extractErrorMessage(response, fallback = "Произошла ошибка") {
    const contentType = response.headers.get("Content-Type") || "";
    if (contentType.includes("application/json")) {
        try {
            const data = await response.json();
            if (data?.message) {
                return data.message;
            }
            if (data?.description) {
                return data.description;
            }
            if (data?.error) {
                return data.error;
            }
        } catch (error) {
            console.debug("error parsing json payload", error);
        }
        return fallback;
    }
    try {
        const text = await response.text();
        return text || fallback;
    } catch (error) {
        console.debug("error reading response body", error);
        return fallback;
    }
}

function formatDate(timestamp) {
    if (!timestamp) {
        return "только что";
    }
    const date = new Date(timestamp * 1000);
    return date.toLocaleString("ru-RU", {
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
    });
}

function formatAuthor(author) {
    if (!author || !author.id) {
        return "Аноним";
    }
    if (author.name) {
        return author.name;
    }
    if (author.username) {
        return `@${author.username}`;
    }
    return `ID ${author.id}`;
}

function showModal() {
    elements.modal.classList.remove("hidden");
    elements.modalError.classList.add("hidden");
    elements.modalInput.value = "";
    setTimeout(() => elements.modalInput.focus(), 50);
}

function hideModal() {
    elements.modal.classList.add("hidden");
}

function showTweakModal(game) {
    state.currentTweakGameId = game.id;
    elements.tweakModal.classList.remove("hidden");
    elements.tweakError.classList.add("hidden");
    elements.tweakTextarea.value = "";
    elements.tweakTitle.textContent = `Попросить доработку (${game.title || "без названия"})`;
    setTimeout(() => elements.tweakTextarea.focus(), 50);
}

function hideTweakModal() {
    state.currentTweakGameId = null;
    elements.tweakModal.classList.add("hidden");
}

function setLoading(loading) {
    state.loading = loading;
    elements.loadMore.disabled = loading || state.reachedEnd;
    if (loading) {
        elements.loadMore.textContent = "Загружаем...";
    } else if (state.reachedEnd) {
        elements.loadMore.textContent = "Это всё";
        elements.loadMore.disabled = true;
    } else {
        elements.loadMore.textContent = "Показать ещё";
    }
}

function startGeneratorProgress(initialMessage = PROGRESS_PHRASES[0]) {
    stopGeneratorProgress();
    if (!elements.generatorStatus) {
        return;
    }
    elements.generatorStatus.classList.remove("error", "success");
    elements.generatorStatus.classList.add("loading");
    elements.generatorStatus.innerHTML = initialMessage;
    const startIndex = PROGRESS_PHRASES.indexOf(initialMessage);
    progressStep = startIndex >= 0 ? startIndex : -1;
    progressIntervalId = window.setInterval(() => {
        if (!elements.generatorStatus) {
            return;
        }
        progressStep = (progressStep + 1) % PROGRESS_PHRASES.length;
        elements.generatorStatus.innerHTML = PROGRESS_PHRASES[progressStep];
    }, 2200);
}

function stopGeneratorProgress() {
    if (progressIntervalId) {
        window.clearInterval(progressIntervalId);
        progressIntervalId = null;
    }
    if (elements.generatorStatus) {
        elements.generatorStatus.classList.remove("loading");
    }
}

function setGenerating(isGenerating, message = "", variant = "info") {
    state.generating = isGenerating;
    if (elements.generatorSubmit) {
        elements.generatorSubmit.disabled = isGenerating;
    }
    if (!elements.generatorStatus) {
        return;
    }
    if (isGenerating) {
        const startingMessage = message || PROGRESS_PHRASES[0];
        startGeneratorProgress(startingMessage);
        return;
    }

    stopGeneratorProgress();
    elements.generatorStatus.classList.remove("error", "success");
    if (message) {
        elements.generatorStatus.innerHTML = message;
    } else {
        elements.generatorStatus.textContent = "";
    }
    if (variant === "error") {
        elements.generatorStatus.classList.add("error");
    } else if (variant === "success") {
        elements.generatorStatus.classList.add("success");
    }
}

function renderSession() {
    elements.sessionControls.innerHTML = "";
    if (state.user) {
        const label = document.createElement("span");
        label.className = "session-label";
        label.textContent = `👋 ${state.user.name || state.user.username || "Безымянный"}`;
        const logoutBtn = document.createElement("button");
        logoutBtn.className = "button ghost";
        logoutBtn.textContent = "Выйти";
        logoutBtn.addEventListener("click", logout);
        elements.sessionControls.append(label, logoutBtn);
    } else {
        const loginBtn = document.createElement("button");
        loginBtn.className = "button ghost";
        loginBtn.id = "login-button";
        loginBtn.textContent = "Войти по коду";
        loginBtn.addEventListener("click", showModal);
        elements.sessionControls.append(loginBtn);
    }
}

function renderEmptyState(scope) {
    const article = document.createElement("article");
    article.className = "empty-state";
    article.innerHTML = scope === "mine"
        ? "Пока что здесь пусто. Сгенерируй игру через форму выше или попроси агента доработать существующую песочницу."
        : "Игры появятся в ближайшее время. Обнови страницу позже или создай новый проект прямо сейчас.";
    elements.gameList.append(article);
}

function canTweakGame(game) {
    if (!state.user) {
        return false;
    }
    if (state.user.is_admin) {
        return true;
    }
    const authorId = game?.author?.id;
    if (authorId === null || authorId === undefined) {
        return false;
    }
    const userId = state.user.id;
    if (userId === null || userId === undefined) {
        return false;
    }
    return String(authorId) === String(userId);
}

function renderGames(games, reset = false) {
    if (reset) {
        elements.gameList.innerHTML = "";
    }
    if (games.length === 0 && state.offset === 0) {
        renderEmptyState(state.scope);
        return;
    }
    for (const game of games) {
        const clone = elements.gameTemplate.content.cloneNode(true);
        clone.querySelector(".game-title").textContent = game.title || "Без названия";
        clone.querySelector(".game-summary").textContent = game.summary || "Описание будет готово позже.";
        clone.querySelector(".game-model").textContent = game.model ? `Модель: ${game.model}` : "";
        clone.querySelector(".game-author").textContent = `Автор: ${formatAuthor(game.author)}`;
        clone.querySelector(".game-time").textContent = formatDate(game.created_at);
        const playLink = clone.querySelector("a.button.primary");
        const shareUrl = game.share_url || `/webapp/sandbox.html?game_id=${encodeURIComponent(game.id)}`;
        playLink.href = shareUrl;
        const copyBtn = clone.querySelector("button[data-copy]");
        copyBtn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(shareUrl);
                copyBtn.textContent = "Скопировано";
                setTimeout(() => (copyBtn.textContent = "Скопировать ссылку"), 1500);
            } catch (err) {
                console.error("copy failed", err);
                copyBtn.textContent = "Не удалось";
                setTimeout(() => (copyBtn.textContent = "Скопировать ссылку"), 1500);
            }
        });
        const tweakBtn = clone.querySelector("button[data-tweak]");
        if (tweakBtn) {
            const userCanTweak = canTweakGame(game);
            if (userCanTweak) {
                tweakBtn.addEventListener("click", () => showTweakModal(game));
            } else if (!state.user) {
                tweakBtn.classList.add("locked");
                tweakBtn.textContent = "Войти, чтобы доработать";
                tweakBtn.title = "Войдите через код из бота, чтобы просить доработку.";
                tweakBtn.addEventListener("click", showModal);
            } else {
                tweakBtn.classList.add("locked");
                tweakBtn.disabled = true;
                tweakBtn.textContent = "Только автор";
                tweakBtn.title = "Только автор игры может запросить изменения.";
            }
        }
        elements.gameList.append(clone);
    }
}

async function fetchSession() {
    try {
        const response = await fetch(routes.authSession, { credentials: "same-origin" });
        if (!response.ok) {
            throw new Error("session request failed");
        }
        const data = await response.json();
        state.user = data.authenticated ? data.user : null;
        renderSession();
    } catch (error) {
        console.error("session error", error);
        state.user = null;
        renderSession();
    }
}

async function loadGames({ reset = false } = {}) {
    if (state.loading || (state.reachedEnd && !reset)) {
        return;
    }
    if (reset) {
        state.offset = 0;
        state.reachedEnd = false;
    }
    setLoading(true);
    if (reset) {
        elements.gameList.innerHTML = "<div class=\"spinner\"></div>";
    }
    const params = new URLSearchParams({ limit: String(state.limit), offset: String(state.offset) });
    if (state.scope === "mine") {
        params.set("scope", "mine");
    }
    const response = await fetch(`${routes.games}?${params.toString()}`);
    if (!response.ok) {
        setLoading(false);
        if (response.status === 401 && state.scope === "mine") {
            state.scope = "all";
            updateChips();
            alert("Нужен код от бота, чтобы увидеть свои игры.");
            await loadGames({ reset: true });
            return;
        }
        console.error("Не удалось получить список игр", response.status);
        return;
    }
    const data = await response.json();
    const items = data.items || [];
    if (reset) {
        elements.gameList.innerHTML = "";
    }
    renderGames(items, false);
    if (items.length < state.limit) {
        state.reachedEnd = true;
    } else {
        state.offset += state.limit;
    }
    setLoading(false);
}

async function login(code) {
    const payload = { code };
    const response = await fetch(routes.authLogin, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        credentials: "same-origin",
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Код не подошёл");
    }
    const data = await response.json();
    state.user = data.user || null;
    renderSession();
}

async function handleGeneratorSubmit(event) {
    event.preventDefault();
    if (state.generating) {
        return;
    }
    const idea = elements.generatorTextarea.value.trim();
    if (!idea) {
        setGenerating(false, "Напиши, во что будем играть.", "error");
        return;
    }
    setGenerating(true, PROGRESS_PHRASES[0]);
    try {
        const response = await fetch(routes.games, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ idea }),
            credentials: "same-origin",
        });
        if (!response.ok) {
            const fallback = response.status === 401
                ? "Нужен код от бота, чтобы создавать игры."
                : "Не удалось сгенерировать игру.";
            const errorMessage = await extractErrorMessage(response, fallback);
            setGenerating(false, errorMessage, "error");
            if (response.status === 401) {
                showModal();
            }
            return;
        }
        const data = await response.json();
        const game = data.game;
        elements.generatorTextarea.value = "";
        const link = game && game.share_url ? `<a href="${game.share_url}">Открыть игру</a>` : "";
        setGenerating(false, `Готово! ${game?.title || "Новая игра"}. ${link}`, "success");
        await loadGames({ reset: true });
    } catch (error) {
        console.error("generate failed", error);
        setGenerating(false, "Не удалось сгенерировать игру. Попробуй уточнить описание и повтори.", "error");
    }
}

async function handleTweakSubmit(event) {
    event.preventDefault();
    if (!state.currentTweakGameId) {
        return;
    }
    const instructions = elements.tweakTextarea.value.trim();
    if (!instructions) {
        elements.tweakError.textContent = "Опиши, что нужно изменить.";
        elements.tweakError.classList.remove("hidden");
        return;
    }
    elements.tweakError.classList.add("hidden");
    const submitButton = elements.tweakForm.querySelector("button[type=submit]");
    submitButton.disabled = true;
    submitButton.textContent = "Применяем...";
    try {
        const response = await fetch(
            `${routes.games}/${encodeURIComponent(state.currentTweakGameId)}${routes.tweakSuffix}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ instructions }),
                credentials: "same-origin",
            },
        );
        if (!response.ok) {
            const fallback = response.status === 401
                ? "Нужен код от бота, чтобы дорабатывать игры."
                : response.status === 403
                    ? "Только автор игры может просить доработку."
                    : "Не удалось обновить игру.";
            const errorMessage = await extractErrorMessage(response, fallback);
            if (response.status === 401) {
                hideTweakModal();
                showModal();
            } else {
                elements.tweakError.textContent = errorMessage;
                elements.tweakError.classList.remove("hidden");
            }
            return;
        }
        hideTweakModal();
        await loadGames({ reset: true });
    } catch (error) {
        console.error("tweak failed", error);
        elements.tweakError.textContent = "Не получилось обновить игру. Попробуй переформулировать запрос.";
        elements.tweakError.classList.remove("hidden");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Применить";
    }
}

async function logout() {
    await fetch(routes.authLogout, { method: "POST", credentials: "same-origin" });
    state.user = null;
    renderSession();
    if (state.scope === "mine") {
        state.scope = "all";
        updateChips();
        await loadGames({ reset: true });
    }
}

function updateChips() {
    elements.chips.forEach((chip) => {
        chip.classList.toggle("active", chip.dataset.scope === state.scope);
    });
}

function handleChipClick(event) {
    const button = event.target.closest("button[data-scope]");
    if (!button) {
        return;
    }
    const nextScope = button.dataset.scope;
    if (nextScope === "mine" && !state.user) {
        showModal();
        return;
    }
    if (state.scope === nextScope) {
        return;
    }
    state.scope = nextScope;
    updateChips();
    loadGames({ reset: true });
}

async function handleLoginSubmit(event) {
    event.preventDefault();
    const rawCode = elements.modalInput.value.trim().toUpperCase();
    if (!rawCode) {
        return;
    }
    elements.modalError.classList.add("hidden");
    elements.modalError.textContent = "";
    const submitButton = elements.modalForm.querySelector("button[type=submit]");
    submitButton.disabled = true;
    submitButton.textContent = "Проверяем...";
    try {
        await login(rawCode);
        hideModal();
        state.scope = "mine";
        updateChips();
        await loadGames({ reset: true });
    } catch (error) {
        console.error("login failed", error);
        elements.modalError.textContent = "Не удалось войти. Проверь код или запроси новый у бота.";
        elements.modalError.classList.remove("hidden");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Войти";
    }
}

function bindEvents() {
    elements.filters.addEventListener("click", handleChipClick);
    elements.loadMore.addEventListener("click", () => loadGames({ reset: false }));
    elements.modalCancel.addEventListener("click", hideModal);
    elements.modalForm.addEventListener("submit", handleLoginSubmit);
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            if (!elements.modal.classList.contains("hidden")) {
                hideModal();
            }
            if (!elements.tweakModal.classList.contains("hidden")) {
                hideTweakModal();
            }
        }
    });
    if (elements.generatorForm) {
        elements.generatorForm.addEventListener("submit", handleGeneratorSubmit);
    }
    if (elements.tweakForm) {
        elements.tweakForm.addEventListener("submit", handleTweakSubmit);
    }
    if (elements.tweakCancel) {
        elements.tweakCancel.addEventListener("click", hideTweakModal);
    }
}

async function bootstrap() {
    bindEvents();
    await fetchSession();
    updateChips();
    await loadGames({ reset: true });
}

bootstrap().catch((error) => {
    console.error("hub bootstrap error", error);
});
