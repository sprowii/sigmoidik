const statusElement = document.getElementById("status");
const containerElement = document.getElementById("game-container");

let cachedPhaser = null;

const GAME_ID_PARAM_NAMES = ["game_id", "gameId", "id"];
const FORBIDDEN_PATTERNS = [
    { pattern: /document\.(cookie|write|location)/i, reason: "доступ к document.cookie/document.write" },
    { pattern: /localStorage/i, reason: "localStorage запрещён" },
    { pattern: /sessionStorage/i, reason: "sessionStorage запрещён" },
    { pattern: /window\.(parent|top|opener)/i, reason: "запрещён доступ к window.parent/top" },
    { pattern: /\beval\s*\(/i, reason: "eval запрещён" },
    { pattern: /new\s+Function\s*\(/i, reason: "new Function запрещён" },
    { pattern: /<script/i, reason: "вставка <script> запрещена" },
];

function setStatus(message, isError = false) {
    if (!statusElement) {
        return;
    }
    statusElement.textContent = message;
    statusElement.classList.toggle("error", isError);
}

function clearStatus() {
    setStatus("", false);
}

function getGameId() {
    const params = new URLSearchParams(window.location.search);
    for (const key of GAME_ID_PARAM_NAMES) {
        const value = params.get(key);
        if (value) {
            return value.trim();
        }
    }
    return null;
}

function validateCodeSafety(code) {
    const issues = FORBIDDEN_PATTERNS.filter(({ pattern }) => pattern.test(code));
    if (issues.length > 0) {
        const reasonList = issues.map((item) => item.reason).join(", ");
        throw new Error(`Код отклонён политикой безопасности: ${reasonList}`);
    }
}

function createSandboxEnvironment() {
    const cleanupCallbacks = [];
    return {
        version: Phaser.VERSION,
        setStatus,
        clearStatus,
        getContainer: () => containerElement,
        registerCleanup: (fn) => {
            if (typeof fn === "function") {
                cleanupCallbacks.push(fn);
            }
        },
        cleanup: () => {
            while (cleanupCallbacks.length > 0) {
                const fn = cleanupCallbacks.shift();
                try {
                    fn?.();
                } catch (error) {
                    console.warn("Ошибка при очистке ресурса:", error);
                }
            }
        },
    };
}

async function getPhaserInstance() {
    if (cachedPhaser) {
        return cachedPhaser;
    }
    if (window.Phaser) {
        cachedPhaser = window.Phaser;
        return cachedPhaser;
    }
    return new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
            reject(new Error("Библиотека Phaser не загрузилась."));
        }, 8000);
        const check = () => {
            if (window.Phaser) {
                clearTimeout(timeoutId);
                cachedPhaser = window.Phaser;
                resolve(cachedPhaser);
            } else {
                requestAnimationFrame(check);
            }
        };
        check();
    });
}

async function wrapAndExecute(code, sandbox) {
    const wrapped = `(async function(Phaser, sandbox) {\n"use strict";\n${code}\n})`;
    let executable;
    try {
        executable = window.Function("Phaser", "sandbox", wrapped);
    } catch (compileError) {
        console.error("Не удалось скомпилировать игру", compileError);
        throw new Error("Игра содержит синтаксическую ошибку.");
    }

    const phaser = await getPhaserInstance();
    sandbox.setStatus(`Phaser v${phaser.VERSION}: запуск игры...`);

    try {
        const result = executable(phaser, sandbox);
        const maybePromise = normalizeExecutionResult(result, sandbox, phaser);
        if (maybePromise && typeof maybePromise.then === "function") {
            maybePromise.catch((error) => {
                console.error("Асинхронная ошибка игры", error);
                setStatus("Игра завершилась с ошибкой.", true);
            });
        }
        sandbox.setStatus("Игра запущена");
    } catch (runtimeError) {
        console.error("Игра завершилась с ошибкой", runtimeError);
        throw new Error(runtimeError?.message || "Игра завершилась с ошибкой.");
    }
}

function normalizeExecutionResult(result, sandbox, phaser) {
    if (typeof result === "function") {
        try {
            const nestedResult = result(phaser, sandbox);
            return nestedResult;
        } catch (error) {
            console.error("Ошибка при запуске возвращённой функции", error);
            throw error;
        }
    }
    return result;
}

async function fetchGamePayload(gameId) {
    const response = await fetch(`/api/games/${encodeURIComponent(gameId)}`, {
        headers: {
            "Accept": "application/json",
        },
    });
    if (!response.ok) {
        const message = `Не удалось загрузить игру (статус ${response.status}).`;
        throw new Error(message);
    }
    return response.json();
}

async function bootstrap() {
    if (!containerElement) {
        setStatus("Контейнер игры не найден.", true);
        return;
    }

    const gameId = getGameId();
    if (!gameId) {
        setStatus("Не найден идентификатор игры в ссылке.", true);
        return;
    }

    setStatus("Загружаем игру...");

    let payload;
    try {
        payload = await fetchGamePayload(gameId);
    } catch (error) {
        console.error("Ошибка загрузки игры", error);
        setStatus(error.message || "Не удалось загрузить игру.", true);
        return;
    }

    const { title, summary, code } = payload;
    if (title) {
        setStatus(`Игра: ${title}`);
    }

    if (typeof code !== "string" || code.trim().length === 0) {
        setStatus("Код игры отсутствует.", true);
        return;
    }

    try {
        validateCodeSafety(code);
    } catch (securityError) {
        console.error("Нарушение политики безопасности", securityError);
        setStatus(securityError.message, true);
        return;
    }

    const sandbox = createSandboxEnvironment();

    try {
        await wrapAndExecute(code, sandbox);
        const startupMessage = summary ? `${summary}` : "Игра сгенерирована и запущена.";
        sandbox.setStatus(startupMessage);
        if (summary) {
            console.info("Описание игры:", summary);
        }
    } catch (executionError) {
        sandbox.cleanup();
        setStatus(executionError.message || "Ошибка при запуске игры.", true);
    }

    window.addEventListener("beforeunload", () => sandbox.cleanup(), { once: true });
}

window.addEventListener("error", (event) => {
    console.error("Global error", event.error || event.message);
    setStatus(`Ошибка: ${event.message}`, true);
});

window.addEventListener("unhandledrejection", (event) => {
    console.error("Unhandled rejection", event.reason);
    const message = event.reason && event.reason.message ? event.reason.message : String(event.reason);
    setStatus(`Необработанное исключение: ${message}`, true);
});

bootstrap().catch((fatalError) => {
    console.error("Критическая ошибка", fatalError);
    const message = fatalError && fatalError.message ? `Критическая ошибка: ${fatalError.message}` : "Критическая ошибка при запуске игры.";
    setStatus(message, true);
});
