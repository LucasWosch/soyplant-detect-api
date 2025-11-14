// Ajuste aqui se o backend estiver em outro host/porta
const apiBase = "http://localhost:8000";

let pc = null;
let localStream = null;
let statsChannel = null;

// Elementos de UI
const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const localOverlay = document.getElementById("localOverlay");
const remoteOverlay = document.getElementById("remoteOverlay");
const localStatus = document.getElementById("localStatus");
const remoteStatus = document.getElementById("remoteStatus");
const sortTotalEl = document.getElementById("sortTotal");
const fpsValueEl = document.getElementById("fpsValue");
const frameIndexEl = document.getElementById("frameIndex");
const logBox = document.getElementById("logBox");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const backBtn = document.getElementById("backBtn");

// Utilitário simples de log
function appendLog(msg) {
    const div = document.createElement("div");
    const ts = new Date().toLocaleTimeString("pt-BR", { hour12: false });
    div.innerHTML = `<span>[${ts}]</span> ${msg}`;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
}

// Atualiza pill de status
function setStatus(elem, text, online) {
    const dot = elem.querySelector(".status-dot");
    const label = elem.querySelector("span:last-child");
    label.textContent = text;
    if (online) {
        elem.classList.add("online");
    } else {
        elem.classList.remove("online");
    }
}

// Iniciar sessão WebRTC
async function start() {
    try {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        appendLog("Iniciando captura da webcam…");

        // 1) captura webcam
        localStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30, max: 60 },
            },
            audio: false,
        });

        localVideo.srcObject = localStream;
        localOverlay.style.display = "none";
        setStatus(localStatus, "Webcam ativa", true);

        // 2) cria RTCPeerConnection
        pc = new RTCPeerConnection({
            iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        });

        appendLog("RTCPeerConnection criado.");

        // 3) adiciona trilhas locais
        localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

        // 4) vídeo remoto
        pc.ontrack = (ev) => {
            appendLog("Track remota recebida.");
            if (ev.streams && ev.streams[0]) {
                remoteVideo.srcObject = ev.streams[0];
            } else {
                const inbound = new MediaStream();
                inbound.addTrack(ev.track);
                remoteVideo.srcObject = inbound;
            }
            remoteOverlay.style.display = "none";
            setStatus(remoteStatus, "Vídeo processado ativo", true);
        };

        // 5) DataChannel vindo do servidor (stats)
        pc.ondatachannel = (event) => {
            const channel = event.channel;
            appendLog(`DataChannel recebido: ${channel.label}`);
            if (channel.label === "stats") {
                statsChannel = channel;
                statsChannel.onopen = () => {
                    appendLog("DataChannel 'stats' aberto.");
                };
                statsChannel.onclose = () => {
                    appendLog("DataChannel 'stats' fechado.");
                };
                statsChannel.onerror = (err) => {
                    appendLog("Erro no DataChannel 'stats': " + err);
                };
                statsChannel.onmessage = (ev) => {
                    try {
                        const data = JSON.parse(ev.data);
                        if (data.type === "stats") {
                            if (typeof data.sort_unique_ids === "number") {
                                sortTotalEl.textContent = data.sort_unique_ids.toString();
                            }
                            if (typeof data.fps === "number") {
                                fpsValueEl.textContent = data.fps.toFixed(1);
                            }
                            if (typeof data.frame_index === "number") {
                                frameIndexEl.textContent = data.frame_index.toString();
                            }
                        }
                    } catch (e) {
                        appendLog("Falha ao parsear stats: " + e);
                    }
                };
            }
        };

        // 6) Estado da conexão
        pc.onconnectionstatechange = () => {
            const st = pc.connectionState;
            appendLog("Estado da conexão: " + st);
            if (st === "failed" || st === "disconnected" || st === "closed") {
                setStatus(remoteStatus, "Conexão perdida", false);
            } else if (st === "connected") {
                setStatus(remoteStatus, "Conectado", true);
            }
        };

        // 7) cria e envia offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        appendLog("Enviando offer para o servidor…");

        const resp = await fetch(`${apiBase}/webrtc/offer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        });

        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`Offer falhou: ${resp.status} - ${text}`);
        }

        const answer = await resp.json();
        await pc.setRemoteDescription(answer);
        appendLog("Answer recebida e aplicada. WebRTC estabelecido.");

    } catch (err) {
        console.error(err);
        appendLog("Erro ao iniciar sessão: " + err);
        await stop();
    }
}

// Parar sessão
async function stop() {
    appendLog("Parando sessão…");

    stopBtn.disabled = true;
    startBtn.disabled = false;

    try {
        if (statsChannel) {
            try {
                statsChannel.close();
            } catch (e) {
                // ignore
            }
            statsChannel = null;
        }

        if (pc) {
            pc.getSenders().forEach((s) => {
                try {
                    if (s.track) s.track.stop();
                } catch (_) {}
            });
            pc.getReceivers().forEach((r) => {
                try {
                    if (r.track) r.track.stop();
                } catch (_) {}
            });
            await pc.close();
            pc = null;
        }

        if (localStream) {
            localStream.getTracks().forEach((t) => t.stop());
            localStream = null;
        }

        localVideo.srcObject = null;
        remoteVideo.srcObject = null;
        localOverlay.style.display = "";
        remoteOverlay.style.display = "";

        setStatus(localStatus, "Offline", false);
        setStatus(remoteStatus, "Aguardando conexão", false);

        sortTotalEl.textContent = "0";
        fpsValueEl.textContent = "0.0";
        frameIndexEl.textContent = "-";

        appendLog("Sessão encerrada.");
    } catch (err) {
        console.error(err);
        appendLog("Erro ao parar sessão: " + err);
    }
}

// Botão voltar (ajuste o href para tua tela principal)
backBtn.addEventListener("click", () => {
    window.location.href = "index.html";
});

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// boa prática: se fechar aba/janela, parar
window.addEventListener("beforeunload", () => {
    if (pc || localStream) {
        // não await aqui
        stop();
    }
});
