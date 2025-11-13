const statusElement = document.getElementById("status");
const frameElement = document.getElementById("game-frame");

const GAME_ID_PARAM_NAMES = ["game_id", "gameId", "id"];

/**
 * Жёсткая фильтрация: несмотря на то, что код запускается в sandbox iframe,
 * режем явные обращения к глобальным объектам и опасным API.
 */
const FORBIDDEN_PATTERNS = [
    // Глобальные объекты
    { pattern: /\bwindow\s*\./i, reason: "доступ к window.* запрещён" },
    { pattern: /\bdocument\s*\./i, reason: "доступ к document.* запрещён" },
    { pattern: /\bparent\s*\./i, reason: "доступ к parent.* запрещён" },
    { pattern: /\btop\s*\./i, reason: "доступ к top.* запрещён" },
    { pattern: /\bopener\s*\./i, reason: "доступ к opener.* запрещён" },

    // Хранилища
    { pattern: /\blocalStorage\b/i, reason: "localStorage запрещён" },
    { pattern: /\bsessionStorage\b/i, reason: "sessionStorage запрещён" },
    { pattern: /\bindexedDB\b/i, reason: "indexedDB запрещён" },

    // Исполнение кода
    { pattern: /\beval\s*\(/i, reason: "eval запрещён" },
    { pattern: /\bnew\s+Function\s*\(/i, reason: "new Function запрещён" },
    { pattern: /\bFunction\s*\(/i, reason: "Function конструктор запрещён" },

    // Воркер/скрипты
    { pattern: /\bimportScripts\s*\(/i, reason: "importScripts запрещён" },
    { pattern: /\bWorker\s*\(/i, reason: "Worker запрещён" },
    { pattern: /\bSharedWorker\s*\(/i, reason: "SharedWorker запрещён" },

    // Вставка скриптов
    { pattern: /<script/i, reason: "вставка <script> запрещена" },
];

function setStatus(message, secondArg = false) {
    if (!statusElement) {
        return;
    }
    let isError = false;
    let loading = false;
    if (typeof secondArg === "object" && secondArg !== null) {
        isError = Boolean(secondArg.error);
        loading = Boolean(secondArg.loading);
    } else {
        isError = Boolean(secondArg);
    }
    statusElement.textContent = message;
    statusElement.classList.toggle("error", isError);
    statusElement.classList.toggle("loading", loading);
}

function getGameId() {
    const params = new URLSearchParams(window.location.search);
    for (const key of GAME_ID_PARAM_NAMES) {
        const value = params.get(key);
        if (value) return value.trim();
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

async function fetchGamePayload(gameId) {
    const response = await fetch(`/api/games/${encodeURIComponent(gameId)}`, {
        headers: {
            "Accept": "application/json",
        },
    });
    if (!response.ok) {
        throw new Error(`Не удалось загрузить игру (статус ${response.status}).`);
    }
    return response.json();
}

/**
 * Обёртка вокруг пользовательского кода.
 *
 * Нейросеть должна вернуть функцию:
 *   - return function run2D(create2D) { ... }
 *   - или return function run3D(create3D) { ... }
 *   - или return function run(create2D, create3D) { ... }
 */
function buildGameWrapper(code) {
    return `
(function () {
    "use strict";

    function initGame(rootElement) {
        const canvas = document.createElement("canvas");
        canvas.id = "game";
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        rootElement.appendChild(canvas);

        const dpr = window.devicePixelRatio || 1;
        const raf = window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); };
        const caf = window.cancelAnimationFrame || function (id) { clearTimeout(id); };

        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            const width = Math.max(1, rect.width * dpr);
            const height = Math.max(1, rect.height * dpr);
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }
        }

        resizeCanvas();
        window.addEventListener("resize", resizeCanvas);

        const commonUtils = {
            random() {
                return Math.random();
            },
            now() {
                return performance.now();
            }
        };

        function create2D() {
            const ctx = canvas.getContext("2d");
            if (!ctx) throw new Error("2D контекст недоступен.");

            const frameHandlers = new Set();
            let frameId = null;

            function loop(timestamp) {
                frameId = raf(loop);
                frameHandlers.forEach((cb) => {
                    try { cb(timestamp); } catch (e) { console.error("Ошибка в onFrame 2D:", e); }
                });
            }

            frameId = raf(loop);

            const utils2D = {
                ...commonUtils,
                onFrame(cb) {
                    if (typeof cb !== "function") return () => {};
                    frameHandlers.add(cb);
                    return () => frameHandlers.delete(cb);
                },
                onResize(cb) {
                    if (typeof cb !== "function") return () => {};
                    function handle() {
                        cb(canvas.width, canvas.height);
                    }
                    window.addEventListener("resize", handle);
                    handle();
                    return () => window.removeEventListener("resize", handle);
                },
                clear() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            };

            return { canvas, ctx, utils: utils2D };
        }

        function create3D() {
            if (!window.THREE) {
                throw new Error("THREE.js не загружен.");
            }

            const THREE = window.THREE;
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x020617);

            const camera = new THREE.PerspectiveCamera(
                60,
                canvas.width / canvas.height,
                0.1,
                100
            );
            camera.position.set(0, 1.5, 4);

            const renderer = new THREE.WebGLRenderer({
                canvas,
                antialias: true,
            });
            renderer.setPixelRatio(dpr);
            renderer.setSize(canvas.width, canvas.height, false);

            const frameHandlers = new Set();
            const clock = new THREE.Clock();

            function loop() {
                raf(loop);
                const elapsed = clock.getElapsedTime();
                frameHandlers.forEach((cb) => {
                    try { cb(elapsed); } catch (e) { console.error("Ошибка в onFrame 3D:", e); }
                });
                renderer.render(scene, camera);
            }

            loop();

            function onResize() {
                const width = canvas.width;
                const height = canvas.height;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height, false);
            }

            window.addEventListener("resize", () => {
                resizeCanvas();
                onResize();
            });
            onResize();

            // === ЗАГРУЗЧИКИ РЕСУРСОВ ===
            const textureLoader = new THREE.TextureLoader();
            let gltfLoader = null;
            if (THREE.GLTFLoader) {
                gltfLoader = new THREE.GLTFLoader();
            }

            const utils3D = {
                ...commonUtils,
                onFrame(cb) {
                    if (typeof cb !== "function") return () => {};
                    frameHandlers.add(cb);
                    return () => frameHandlers.delete(cb);
                },
                addAmbientLight(intensity) {
                    const light = new THREE.AmbientLight(0xffffff, intensity ?? 0.3);
                    scene.add(light);
                    return light;
                },
                addDirectionalLight(intensity, position) {
                    const light = new THREE.DirectionalLight(0xffffff, intensity ?? 1);
                    if (position && typeof position.x === "number") {
                        light.position.set(position.x, position.y, position.z);
                    } else {
                        light.position.set(3, 5, 2);
                    }
                    scene.add(light);
                    return light;
                },
                async loadTexture(url) {
                    return new Promise((resolve, reject) => {
                        try {
                            textureLoader.load(
                                url,
                                (texture) => resolve(texture),
                                undefined,
                                (err) => reject(err || new Error("Не удалось загрузить текстуру"))
                            );
                        } catch (e) {
                            reject(e);
                        }
                    });
                },
                async loadModel(url) {
                    if (!gltfLoader) {
                        throw new Error("GLTFLoader недоступен.");
                    }
                    return new Promise((resolve, reject) => {
                        try {
                            gltfLoader.load(
                                url,
                                (gltf) => {
                                    resolve(gltf.scene || gltf.scenes?.[0] || gltf);
                                },
                                undefined,
                                (err) => reject(err || new Error("Не удалось загрузить модель"))
                            );
                        } catch (e) {
                            reject(e);
                        }
                    });
                }
            };

            return { THREE, scene, camera, renderer, utils: utils3D };
        }

        // === Пользовательский код: должен вернуть функцию ===
        const runFactory = (function () {
${code}
        }());

        if (typeof runFactory === "function") {
            try {
                // Вариант 1: принимает оба create2D и create3D
                if (runFactory.length >= 2) {
                    runFactory(create2D, create3D);
                    return;
                }
                // Вариант 2: принимает один аргумент — считаем, что 3D, если имя содержит "3D"
                if (runFactory.length === 1) {
                    const name = runFactory.name || "";
                    if (/3d/i.test(name)) {
                        runFactory(create3D);
                    } else {
                        runFactory(create2D);
                    }
                    return;
                }
                // Вариант 3: без аргументов — передаём оба, пусть сам решает
                runFactory(create2D, create3D);
            } catch (err) {
                console.error("Ошибка при запуске игры:", err);
            }
        } else {
            console.warn("Пользовательский код не вернул функцию для запуска игры.");
        }
    }

    window.__SIGMOIDA_START__ = initGame;
}());
`;
}

/**
 * Строим HTML для srcdoc iframe:
 * - подключаем THREE.js и GLTFLoader;
 * - создаём root-контейнер;
 * - вставляем обёрнутый код;
 * - автозапускаем __SIGMOIDA_START__(root).
 */
function buildFrameSrcDoc(gameCodeWrapped) {
    const escapedCode = gameCodeWrapped.replace(/<\/script/gi, "<\\/script");

    return `
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Sigmoida Game</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            background: #020617;
            color: #e5e7eb;
            height: 100%;
            overflow: hidden;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
        }
        #root {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: stretch;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.161.0/examples/js/loaders/GLTFLoader.js"></script>
    <script>
${escapedCode}
    </script>
    <script>
        (function () {
            "use strict";
            window.addEventListener("load", function () {
                const root = document.getElementById("root");
                if (!root || typeof window.__SIGMOIDA_START__ !== "function") {
                    console.error("Инициализация игры невозможна: нет root или стартовой функции.");
                    return;
                }
                try {
                    window.__SIGMOIDA_START__(root);
                } catch (err) {
                    console.error("Ошибка при запуске игры:", err);
                }
            });
        }());
    </script>
</body>
</html>
`;
}

async function bootstrap() {
    if (!frameElement) {
        setStatus("Контейнер iframe не найден.", true);
        return;
    }

    const gameId = getGameId();
    if (!gameId) {
        setStatus("Не найден идентификатор игры в ссылке.", true);
        return;
    }

    setStatus("Загружаем игру...", { loading: true });

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
        setStatus(`Игра: ${title}`, { loading: true });
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

    let wrapped;
    try {
        wrapped = buildGameWrapper(code);
    } catch (wrapError) {
        console.error("Ошибка при обёртке кода игры", wrapError);
        setStatus("Код игры не удалось подготовить к запуску.", true);
        return;
    }

    const srcdoc = buildFrameSrcDoc(wrapped);
    frameElement.srcdoc = srcdoc;

    if (summary) {
        setStatus(summary);
    } else if (title) {
        setStatus(`Игра «${title}» запущена.`);
    } else {
        setStatus("Игра запущена.");
    }
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
    const message = fatalError && fatalError.message
        ? `Критическая ошибка: ${fatalError.message}`
        : "Критическая ошибка при запуске игры.";
    setStatus(message, true);
});