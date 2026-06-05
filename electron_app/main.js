const { app, BrowserWindow, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

let mainWindow;
let backendProcess;

const PYTHON = process.env.PYTHON || "python";
const BACKEND_PORT = process.env.CLF_PORT || 5200;
const BACKEND_HOST = "127.0.0.1";
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const STARTUP_TIMEOUT = 15000;

function startBackend() {
  const serverScript = path.join(__dirname, "server.py");

  console.log(`[ELECTRON] Starting Python backend: ${PYTHON} ${serverScript}`);
  backendProcess = spawn(PYTHON, [serverScript, "--port", String(BACKEND_PORT), "--host", BACKEND_HOST], {
    cwd: __dirname,
    stdio: ["ignore", "pipe", "pipe"],
  });

  backendProcess.stdout.on("data", (data) => {
    const msg = data.toString().trim();
    console.log(`[BACKEND] ${msg}`);
  });
  backendProcess.stderr.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) console.log(`[BACKEND] ${msg}`);
  });
  backendProcess.on("close", (code) => {
    console.log(`[BACKEND] Process exited with code ${code}`);
  });

  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      const http = require("http");
      const req = http.get(`${BACKEND_URL}/api/state`, (res) => {
        if (res.statusCode === 200) {
          console.log("[ELECTRON] Backend ready");
          resolve();
        } else {
          retry();
        }
      });
      req.on("error", retry);
      req.setTimeout(1000, () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() - start > STARTUP_TIMEOUT) {
        reject(new Error("Backend startup timeout"));
        return;
      }
      setTimeout(check, 400);
    };
    check();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: "CLF Neuro Monitor",
    backgroundColor: "#1e1e2e",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    autoHideMenuBar: true,
    show: false,
  });

  mainWindow.loadURL(BACKEND_URL);
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => { mainWindow = null; });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

app.whenReady().then(async () => {
  try {
    await startBackend();
  } catch (e) {
    console.error("[ELECTRON] Failed to start backend:", e.message);
    app.quit();
    return;
  }
  createWindow();
});

app.on("window-all-closed", () => {
  if (backendProcess) {
    console.log("[ELECTRON] Terminating backend...");
    backendProcess.kill("SIGTERM");
  }
  app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
