// Variáveis globais
let currentUser = null;
let token = null;
let currentLocation = null;

// Variáveis para paginação
let currentPage = 1;
const itemsPerPage = 10;
let totalHistoryItems = 0;
let allHistoryItems = [];
let filteredHistoryItems = [];

// Variáveis Globais WebRTC

const apiBase = "http://localhost:8000";
let localStream = null;
let peerConnection = null;

// ===================================================================
// FUNÇÕES DE TEMA (CORRIGIDAS)
// ===================================================================

function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcons(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcons(newTheme);
}

function updateThemeIcons(theme) {
    const themeToggle = document.getElementById('themeToggle');
    const themeToggleAuth = document.getElementById('themeToggleAuth');
    const themeToggleAuthRegister = document.getElementById('themeToggleAuthRegister');

    const icon = theme === 'dark' ? 'fa-moon' : 'fa-sun';

    if (themeToggle) themeToggle.innerHTML = `<i class="fas ${icon}"></i>`;
    if (themeToggleAuth) themeToggleAuth.innerHTML = `<i class="fas ${icon}"></i>`;
    if (themeToggleAuthRegister) themeToggleAuthRegister.innerHTML = `<i class="fas ${icon}"></i>`;
}

// ===================================================================
// FUNÇÕES DE AUTENTICAÇÃO (LOGIN, REGISTO, LOGOUT)
// ===================================================================

async function login() {
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;

    if (!username || !password) {
        showNotification('Por favor, preencha todos os campos.', 'error');
        return;
    }

    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    try {
        showNotification('A autenticar...', 'info');

        const response = await fetch('http://localhost:8000/api/v1/login', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            token = data.access_token;
            localStorage.setItem('authToken', token);
            localStorage.setItem('username', username);

            currentUser = username;

            document.getElementById('userName').textContent = username;
            document.getElementById('userAvatar').textContent = username.charAt(0).toUpperCase();
            document.getElementById('authContainer').style.display = 'none';
            document.getElementById('mainApp').style.display = 'block';
            showNotification('Acesso concedido. Bem-vindo ao TCC.', 'success');
            autofillLocation();
            loadHistory();
            loadDashboardStats();
        } else {
            const error = await response.json();
            showNotification('Falha na autenticação: ' + (error.detail || 'Credenciais inválidas'), 'error');
        }
    } catch (error) {
        console.error('Erro de rede:', error);
        showNotification('Erro de rede. A API está online?', 'error');
    }
}

function showLoginForm() {
    document.getElementById('loginForm').style.display = 'block';
    document.getElementById('registerForm').style.display = 'none';
}

function showRegisterForm() {
    document.getElementById('loginForm').style.display = 'none';
    document.getElementById('registerForm').style.display = 'block';
}

async function register() {
    const username = document.getElementById('regUsername').value;
    const email = document.getElementById('regEmail').value;
    const password = document.getElementById('regPassword').value;

    if (!username || !email || !password) {
        showNotification('Por favor, complete todos os campos.', 'error');
        return;
    }

    try {
        showNotification('A criar conta...', 'info');
        const response = await fetch('http://localhost:8000/api/v1/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });

        if (response.status === 201) {
            showNotification('Conta criada com sucesso! Já pode fazer o login.', 'success');
            showLoginForm();
            document.getElementById('regUsername').value = '';
            document.getElementById('regEmail').value = '';
            document.getElementById('regPassword').value = '';
        } else {
            const error = await response.json();
            showNotification('Erro ao criar conta: ' + (error.detail || 'Erro desconhecido'), 'error');
        }
    } catch (error) {
        showNotification('Erro de rede. Por favor, tente novamente.', 'error');
    }
}

function logout() {
    try {
        document.getElementById('loginUsername').value = '';
        document.getElementById('loginPassword').value = '';
    } catch (e) {
        console.error("Erro ao limpar campos de login:", e);
    }

    currentUser = null;
    token = null;
    localStorage.removeItem('authToken');
    localStorage.removeItem('username');

    try {
        document.getElementById('totalAnalises').textContent = '0';
        document.getElementById('totalPlantas').textContent = '0';
        document.getElementById('mediaPlantas').textContent = '0';

        document.getElementById('historyList').innerHTML =
            '<p style="text-align: center; color: var(--gray); padding: 3rem;">Nenhum histórico de análise encontrado.</p>';
        document.getElementById('paginationContainer').style.display = 'none';
        allHistoryItems = [];
        filteredHistoryItems = [];
        currentPage = 1;

        const uploadArea = document.querySelector('.upload-area-ultra');
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="upload-icon-ultra">
                    <i class="fas fa-cloud-upload-alt"></i>
                </div>
                <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">Arraste o Vídeo para Análise</h3>
                <p style="color: var(--gray);">Resolução 4K • Máx. 500MB • Todos os Formatos</p>
                <p style="color: var(--primary-light); font-size: 0.9rem; margin-top: 0.5rem;">
                    <i class="fas fa-robot"></i> Potenciado pela Rede Neural YOLOv8
                </p>
            `;
        }

        if (document.getElementById('fileInput')) document.getElementById('fileInput').value = null;
        if (document.getElementById('analyzeBtn')) document.getElementById('analyzeBtn').disabled = true;
        if (document.getElementById('resultsContainer')) document.getElementById('resultsContainer').style.display = 'none';

        showTab('analyze', null);

    } catch (e) {
        console.error("Erro ao resetar a UI durante o logout:", e);
    }

    document.getElementById('authContainer').style.display = 'flex';
    document.getElementById('mainApp').style.display = 'none';
    showNotification('Sessão terminada com sucesso.', 'success');
}

function setupPasswordToggle() {
    const loginToggle = document.getElementById('loginPasswordToggle');
    const loginPassword = document.getElementById('loginPassword');
    const regToggle = document.getElementById('regPasswordToggle');
    const regPassword = document.getElementById('regPassword');

    if (loginToggle && loginPassword) {
        loginToggle.addEventListener('click', function () {
            const type = loginPassword.getAttribute('type') === 'password' ? 'text' : 'password';
            loginPassword.setAttribute('type', type);
            this.innerHTML = type === 'password'
                ? '<i class="fas fa-eye"></i>'
                : '<i class="fas fa-eye-slash"></i>';
        });
    }

    if (regToggle && regPassword) {
        regToggle.addEventListener('click', function () {
            const type = regPassword.getAttribute('type') === 'password' ? 'text' : 'password';
            regPassword.setAttribute('type', type);
            this.innerHTML = type === 'password'
                ? '<i class="fas fa-eye"></i>'
                : '<i class="fas fa-eye-slash"></i>';
        });
    }
}

// Funções de Geolocalização

function autofillLocation() {
    if (!navigator.geolocation) {
        showNotification('Geolocalização não é suportada pelo seu navegador.', 'error');
        return;
    }

    showNotification('A obter a sua localização GPS...', 'info');

    navigator.geolocation.getCurrentPosition(
        async (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            document.getElementById('latitude').value = lat.toFixed(6);
            document.getElementById('longitude').value = lon.toFixed(6);

            try {
                const locationName = await getLocationName(lat, lon);
                const localText = locationName || `Localização: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
                document.getElementById('local').value = localText;

                currentLocation = {
                    latitude: lat,
                    longitude: lon,
                    local_texto: localText
                };

                showNotification('Localização GPS capturada com sucesso!', 'success');
            } catch (error) {
                const localText = `Localização: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
                document.getElementById('local').value = localText;
                currentLocation = {
                    latitude: lat,
                    longitude: lon,
                    local_texto: localText
                };
                showNotification('Coordenadas GPS capturadas!', 'info');
            }
        },
        (error) => {
            let errorMessage = 'Não foi possível obter a sua localização.';
            if (error.code === error.PERMISSION_DENIED) {
                errorMessage = 'Você precisa permitir o acesso à localização no seu navegador.';
            }
            showNotification(errorMessage, 'error');
        },
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 }
    );
}

async function getLocationName(lat, lon) {
    try {
        const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&addressdetails=1`
        );
        const data = await response.json();
        if (data && data.address) {
            const address = data.address;
            let enderecoCompleto = [
                address.road,
                address.house_number,
                address.suburb,
                address.city,
                address.state
            ]
                .filter(Boolean)
                .join(', ');

            return enderecoCompleto || `Localização: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
        }
    } catch (error) {
        console.error('Erro ao obter nome da localização:', error);
    }
    return `Localização: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
}

// Funções de Navegação

function showTab(tabName, element) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));

    document.getElementById(tabName + 'Tab').classList.add('active');

    if (element) {
        element.classList.add('active');
    } else {
        const firstTab = document.querySelector('.tab');
        if (firstTab) {
            firstTab.classList.add('active');
        }
    }

    if (tabName === 'history') loadHistory();
    else if (tabName === 'dashboard') loadDashboardStats();
}

// Funções de Upload de Vídeo

function handleFileSelect(files) {
    if (files.length > 0) {
        const file = files[0];
        const uploadArea = document.querySelector('.upload-area-ultra');
        uploadArea.innerHTML = `
            <div class="upload-icon-ultra">
                <i class="fas fa-file-video"></i>
            </div>
            <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">${file.name}</h3>
            <p style="color: var(--gray);">Tamanho: ${(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            <p style="color: var(--primary-light); font-size: 0.9rem; margin-top: 0.5rem;">
                <i class="fas fa-check"></i> Pronto para análise
            </p>
        `;
        document.getElementById('analyzeBtn').disabled = false;
    }
}

async function analyzeVideo() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        showNotification('Por favor, selecione um arquivo de vídeo primeiro.', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('latitude', document.getElementById('latitude').value || '');
    formData.append('longitude', document.getElementById('longitude').value || '');
    formData.append('local_texto', document.getElementById('local').value || '');

    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';

    showNotification('Processando o vídeo... Isto pode demorar.', 'info');

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 5;
        if (progress >= 95) clearInterval(progressInterval);
        progressBar.style.width = progress + '%';
    }, 800);

    try {
        const response = await fetch('http://localhost:8000/api/v1/analisar-video', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken')}` },
            body: formData
        });

        clearInterval(progressInterval);
        progressBar.style.width = '100%';

        if (response.ok) {
            const result = await response.json();

            if (!result.nome_arquivo_original) {
                result.nome_arquivo_original = file.name;
            }

            showResults(result);
            loadHistory();
            loadDashboardStats();
            showNotification('Análise concluída com sucesso!', 'success');
        } else if (response.status === 401) {
            showNotification('Sessão expirada. Faça login novamente.', 'error');
            logout();
        } else {
            const error = await response.json();
            showNotification('Falha na análise: ' + (error.detail || 'Erro desconhecido'), 'error');
        }
    } catch (error) {
        clearInterval(progressInterval);
        showNotification('Erro de rede durante a análise.', 'error');
    } finally {
        setTimeout(() => {
            progressContainer.style.display = 'none';
        }, 2000);
    }
}

function showResults(result) {
    document.getElementById('resultsContainer').style.display = 'block';
    document.getElementById('plantCount').textContent = result.contagem_total_unicos;

    const resultsContent = document.getElementById('resultsContent');
    resultsContent.innerHTML = `
        <div class="fade-in-up">
            <div style="background: var(--glass); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(0, 212, 170, 0.3);">
                <h4 style="color: var(--primary); margin-bottom: 1rem; font-size: 1.3rem;">
                    <i class="fas fa-check-circle"></i> ${result.message}
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem;">
                    <div>
                        <div style="font-weight: 600; color: var(--gray); margin-bottom: 0.5rem;">Ficheiro</div>
                        <div>${result.nome_arquivo_original}</div>
                    </div>
                    <div>
                        <div style="font-weight: 600; color: var(--gray); margin-bottom: 0.5rem;">Plantas Detectadas</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: var(--primary);">${result.contagem_total_unicos}</div>
                    </div>
                    <div>
                        <div style="font-weight: 600; color: var(--gray); margin-bottom: 0.5rem;">Localização</div>
                        <div>${document.getElementById('local').value || 'Não informada'}</div>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
}

// Funções de Histórico e Paginação

async function loadHistory() {
    if (!localStorage.getItem('authToken')) return;

    try {
        const response = await fetch('http://localhost:8000/api/v1/historico', {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken')}` }
        });

        if (response.ok) {
            const history = await response.json();
            allHistoryItems = history;
            filteredHistoryItems = [...allHistoryItems];
            totalHistoryItems = filteredHistoryItems.length;
            currentPage = 1;
            displayHistory(filteredHistoryItems);
            updatePagination();
        } else if (response.status === 401) {
            showNotification('Sessão expirada. Faça login novamente.', 'error');
            logout();
        } else {
            console.error('Erro ao buscar histórico:', response.status);
            document.getElementById('historyList').innerHTML =
                '<p style="text-align: center; color: var(--accent); padding: 3rem;">Erro ao carregar histórico.</p>';
        }
    } catch (error) {
        console.error('Erro de rede ao carregar histórico:', error);
        document.getElementById('historyList').innerHTML =
            '<p style="text-align: center; color: var(--accent); padding: 3rem;">Erro de rede ao carregar histórico.</p>';
    }
}

function filterHistory() {
    const searchTerm = document.getElementById('historySearch').value.toLowerCase();

    if (!searchTerm) {
        filteredHistoryItems = [...allHistoryItems];
    } else {
        filteredHistoryItems = allHistoryItems.filter(item => {
            const fileName = item.nome_arquivo_original?.toLowerCase() || '';
            const location = item.local_texto?.toLowerCase() || '';
            return fileName.includes(searchTerm) || location.includes(searchTerm);
        });
    }

    currentPage = 1;
    totalHistoryItems = filteredHistoryItems.length;
    displayHistory(filteredHistoryItems);
    updatePagination();
}

function filterByDate() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    if (!startDate && !endDate) {
        filteredHistoryItems = [...allHistoryItems];
    } else {
        filteredHistoryItems = allHistoryItems.filter(item => {
            if (!item.data_analise) return false;
            const itemDate = new Date(item.data_analise);
            const start = startDate ? new Date(startDate) : null;
            const end = endDate ? new Date(endDate) : null;

            if (end) end.setDate(end.getDate() + 1);

            const isAfterStart = !start || itemDate >= start;
            const isBeforeEnd = !end || itemDate < end;

            return isAfterStart && isBeforeEnd;
        });
    }

    currentPage = 1;
    totalHistoryItems = filteredHistoryItems.length;
    displayHistory(filteredHistoryItems);
    updatePagination();

    if (filteredHistoryItems.length === 0) {
        showNotification('Nenhum resultado encontrado para o período selecionado.', 'info');
    }
}

function clearDateFilter() {
    document.getElementById('startDate').value = '';
    document.getElementById('endDate').value = '';
    filteredHistoryItems = [...allHistoryItems];
    currentPage = 1;
    totalHistoryItems = filteredHistoryItems.length;
    displayHistory(filteredHistoryItems);
    updatePagination();
}

function displayHistory(history) {
    const historyList = document.getElementById('historyList');

    if (history.length === 0) {
        historyList.innerHTML =
            '<p style="text-align: center; color: var(--gray); padding: 3rem;">Nenhum histórico de análise encontrado.</p>';
        document.getElementById('paginationContainer').style.display = 'none';
        return;
    }

    history.sort((a, b) => new Date(b.data_analise) - new Date(a.data_analise));

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, history.length);
    const currentPageItems = history.slice(startIndex, endIndex);

    historyList.innerHTML = currentPageItems.map(item => `
        <div class="history-item-ultra fade-in-up">
            <div>
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    ${item.nome_arquivo_original}
                </div>
                <div style="color: var(--gray); font-size: 0.9rem;">
                    <i class="fas fa-calendar"></i>
                    ${item.data_analise ? new Date(item.data_analise).toLocaleString('pt-BR') : 'Data Indisponível'}
                </div>
                ${
                    item.local_texto
                        ? `
                    <div style="color: var(--primary-light); font-size: 0.9rem; margin-top: 0.3rem;">
                        <i class="fas fa-map-marker-alt"></i> ${item.local_texto}
                        ${
                            item.latitude && item.longitude
                                ? ` (${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)})`
                                : ''
                        }
                    </div>
                `
                        : (item.latitude && item.longitude
                            ? `
                    <div style="color: var(--primary-light); font-size: 0.9rem; margin-top: 0.3rem;">
                        <i class="fas fa-map-marker-alt"></i> Coords: ${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)}
                    </div>
                `
                            : '')
                }
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: 800; color: var(--primary); margin-bottom: 0.5rem;">
                    ${item.contagem_total_unicos ?? '?'}
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <div style="background: var(--primary); color: white; padding: 0.3rem 1rem; border-radius: 15px; font-size: 0.8rem; display: inline-block;">
                        Concluído
                    </div>
                </div>

                <button
                    class="btn btn-header"
                    style="margin-top: 0.3rem;"
                    onclick="openHistoryVideoModal(${item.id})"
                >
                    <i class="fas fa-play-circle"></i> Ver vídeo
                </button>
            </div>
        </div>
    `).join('');

    document.getElementById('paginationContainer').style.display =
        totalHistoryItems > itemsPerPage ? 'flex' : 'none';
}

let currentHistoryVideoUrl = null;

async function openHistoryVideoModal(analiseId) {
    const token = localStorage.getItem('authToken');
    if (!token) {
        showNotification('Erro: você precisa estar logado para ver o vídeo.', 'error');
        return;
    }

    // Remove modal anterior se existir
    const existing = document.getElementById('historyVideoModal');
    if (existing) {
        existing.remove();
        if (currentHistoryVideoUrl) {
            URL.revokeObjectURL(currentHistoryVideoUrl);
            currentHistoryVideoUrl = null;
        }
    }

    const modalHTML = `
        <div class="realtime-modal" id="historyVideoModal">
            <div class="realtime-content">
                <div class="realtime-header">
                    <h2 class="realtime-title">
                        <i class="fas fa-play-circle"></i> Vídeo da Análise
                    </h2>
                    <button class="realtime-close" onclick="closeHistoryVideoModal()">×</button>
                </div>

                <div style="margin-top: 1rem;">
                    <div id="historyVideoStatus" class="camera-status" style="justify-content: center;">
                        <i class="fas fa-sync fa-spin camera-icon"></i>
                        <p>Carregando vídeo processado...</p>
                    </div>

                    <video
                        id="historyVideoPlayer"
                        class="video-element"
                        style="display: none; max-height: 70vh; width: 100%; border-radius: 16px;"
                        controls
                    ></video>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);

    const statusEl = document.getElementById('historyVideoStatus');
    const videoEl = document.getElementById('historyVideoPlayer');

    const url = `${apiBase}/api/v1/historico/${analiseId}/video`;
    console.log("[History Video] Requisitando:", url);

    try {
        const resp = await fetch(url, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        console.log("[History Video] Status:", resp.status);
        console.log("[History Video] Content-Type (header):", resp.headers.get('content-type'));

        if (!resp.ok) {
            const txt = await resp.text();
            console.error('[History Video] Erro ao obter vídeo:', resp.status, txt);
            statusEl.innerHTML = `
                <i class="fas fa-exclamation-circle camera-icon"></i>
                <p>Não foi possível carregar o vídeo (${resp.status}).</p>
            `;
            return;
        }

        const blob = await resp.blob();

        console.log("[History Video] Blob size:", blob.size);
        console.log("[History Video] Blob type:", blob.type);



        const objectUrl = URL.createObjectURL(blob);
        currentHistoryVideoUrl = objectUrl;


        // Atualiza o vídeo
        videoEl.pause();
        videoEl.removeAttribute('src'); // garante reset
        videoEl.load();

        videoEl.src = objectUrl;
        videoEl.style.display = 'block';
        statusEl.style.display = 'none';

        // Força o load antes do play
        videoEl.load();

        videoEl.addEventListener('error', (e) => {
            console.error("[History Video] Erro no elemento <video>:", e, videoEl.error);
        }, { once: true });

        videoEl.play().catch(err => {
            console.warn("[History Video] Autoplay bloqueado ou erro ao dar play:", err);
        });

    } catch (err) {
        console.error('[History Video] Erro inesperado:', err);
        statusEl.innerHTML = `
            <i class="fas fa-exclamation-circle camera-icon"></i>
            <p>Erro inesperado ao carregar o vídeo.</p>
        `;
    }
}

function closeHistoryVideoModal() {
    const modal = document.getElementById('historyVideoModal');
    if (modal) {
        modal.remove();
    }
    if (currentHistoryVideoUrl) {
        URL.revokeObjectURL(currentHistoryVideoUrl);
        currentHistoryVideoUrl = null;
    }
}

function changePage(direction) {
    const totalPages = Math.ceil(totalHistoryItems / itemsPerPage);
    const newPage = currentPage + direction;

    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        displayHistory(filteredHistoryItems);
        updatePagination();
    }
}

function updatePagination() {
    const totalPages = Math.ceil(totalHistoryItems / itemsPerPage);
    const pageInfo = document.getElementById('pageInfo');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');

    if (!pageInfo || !prevBtn || !nextBtn) return;

    pageInfo.textContent = `Página ${currentPage} de ${totalPages || 1}`;
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === totalPages || totalPages === 0;

    document.getElementById('paginationContainer').style.display =
        totalPages > 1 ? 'flex' : 'none';
}

// Funções de Dashboard

async function loadDashboardStats() {
    if (!localStorage.getItem('authToken')) return;

    try {
        const response = await fetch('http://localhost:8000/api/v1/historico', {
            headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken')}` }
        });

        if (response.ok) {
            const history = await response.json();
            updateDashboardStats(history);
        } else if (response.status === 401) {
            logout();
        }
    } catch (error) {
        console.error('Erro ao carregar estatísticas:', error);
    }
}

function updateDashboardStats(history) {
    const totalAnalises = history.length;
    const totalPlantas = history.reduce((sum, item) => sum + (item.contagem_total_unicos || 0), 0);
    const mediaPlantas = totalAnalises > 0 ? (totalPlantas / totalAnalises).toFixed(0) : 0;

    document.getElementById('totalAnalises').textContent = totalAnalises;
    document.getElementById('totalPlantas').textContent = totalPlantas;
    document.getElementById('mediaPlantas').textContent = mediaPlantas;
}

// ==============================
// Funções de Detecção em Tempo Real (WebRTC)
// ==============================

peerConnection = null;
localStream = null;
// Garanta que apiBase esteja definido em algum lugar do seu app.js, por exemplo:
// const apiBase = "http://localhost:8000";

function startRealtimeDetection() {
    if (!localStorage.getItem('authToken')) {
        showNotification('Erro: Você precisa estar logado para usar o tempo real.', 'error');
        return;
    }

    const modalHTML = `
        <div class="realtime-modal" id="realtimeModal">
            <div class="realtime-content">
                <div class="realtime-header">
                    <h2 class="realtime-title">
                        <i class="fas fa-video"></i> Deteção em Tempo Real (WebRTC)
                    </h2>
                    <button class="realtime-close" onclick="closeRealtimeDetection()">×</button>
                </div>
                <div class="controls-grid">
                    <button id="startCameraBtn" class="btn btn-primary" onclick="startCamera()">
                        <i class="fas fa-camera"></i> Iniciar Câmera
                    </button>
                    <button id="stopCameraBtn" class="btn btn-header" onclick="stopCamera()" disabled>
                        <i class="fas fa-stop"></i> Parar Câmera
                    </button>
                </div>
                <div class="video-comparison-container">
                    <div class="video-panel">
                        <div class="panel-title">
                            <i class="fas fa-camera"></i> CÂMERA AO VIVO
                        </div>
                        <video id="webcamVideo" class="video-element" autoplay playsinline muted></video>
                        <div id="cameraStatus" class="camera-status">
                            <i class="fas fa-camera camera-icon"></i>
                            <p>Câmera não iniciada</p>
                        </div>
                    </div>
                    <div class="video-panel">
                        <div class="panel-title">
                            <i class="fas fa-robot"></i> DETEÇÃO YOLO (WebRTC)
                        </div>
                        <video id="processedVideo" class="video-element" autoplay playsinline></video>
                        <div id="processingStatus" class="camera-status">
                            <i class="fas fa-sync fa-spin camera-icon"></i>
                            <p>Aguardando processamento</p>
                        </div>
                    </div>
                </div>
                <div class="stats-container">
                    <div id="detectionStats" class="detection-stats">
                        <i class="fas fa-circle" style="color: var(--gray);"></i> Sistema parado
                    </div>
                    <div id="locationStats" class="detection-stats" style="margin-top: 0.5rem; display: none;">
                        <i class="fas fa-map-marker-alt"></i> <span id="locationText">Carregando localização...</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    if (!document.getElementById('realtimeModal')) {
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        const localTexto = document.getElementById('local')?.value || currentLocation?.local_texto;
        if (localTexto) {
            document.getElementById('locationStats').style.display = 'block';
            document.getElementById('locationText').textContent = localTexto;
        } else {
            document.getElementById('locationText').textContent = 'Localização não definida';
        }
    }
}

function closeRealtimeDetection() {
    console.log('[WebRTC] Fechando modal...');
    const modal = document.getElementById('realtimeModal');
    if (modal) {
        stopCamera();
        modal.remove();

        console.log('[UI] Recarregando histórico e dashboard...');
        loadHistory();
        loadDashboardStats();
        showNotification('Sessão em tempo real salva no histórico!', 'success');
    }
}

async function startCamera() {
    const authToken = localStorage.getItem('authToken');
    if (!authToken) {
        showNotification('Erro: Token não encontrado. Faça login.', 'error');
        closeRealtimeDetection();
        return;
    }

    try {
        showNotification('Iniciando câmera...', 'info');

        // Garante que não há conexão antiga
        stopCamera();

        // 1) Captura da webcam
        localStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30, max: 60 }
            },
            audio: false
        });

        const video = document.getElementById('webcamVideo');
        const remoteVideo = document.getElementById('processedVideo');
        const cameraStatus = document.getElementById('cameraStatus');
        const processingStatus = document.getElementById('processingStatus');
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');

        if (!video || !remoteVideo || !cameraStatus || !processingStatus || !startBtn || !stopBtn) {
            console.error('[WebRTC] Elementos de UI não encontrados.');
            return;
        }

        // Mostra vídeo local
        video.srcObject = localStream;
        video.style.display = 'block';
        remoteVideo.style.display = 'block';
        cameraStatus.style.display = 'none';
        processingStatus.style.display = 'none';
        startBtn.disabled = true;
        stopBtn.disabled = false;

        // 2) Conecta via WebRTC
        await connectWebRTC(localStream);

    } catch (error) {
        console.error('Erro ao aceder à câmera:', error);
        let errorMsg = 'Erro ao aceder à câmera. ';
        if (error.name === 'NotAllowedError') errorMsg += 'Verifique as permissões.';
        else if (error.name === 'NotFoundError') errorMsg += 'Nenhuma câmera encontrada.';
        showNotification(errorMsg, 'error');
        stopCamera();
    }
}

async function connectWebRTC(stream) {
    try {
        // 1) Cria PeerConnection com STUN
        peerConnection = new RTCPeerConnection({
            iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        });
        console.log('[WebRTC] Peer Connection criada.');

        // 2) Receber vídeo processado do servidor
        peerConnection.ontrack = (event) => {
            console.log('[WebRTC] Track remota recebida (vídeo processado)!');
            const remoteVideo = document.getElementById('processedVideo');

            if (!remoteVideo) return;

            if (event.streams && event.streams[0]) {
                remoteVideo.srcObject = event.streams[0];
            } else {
                const inbound = new MediaStream();
                inbound.addTrack(event.track);
                remoteVideo.srcObject = inbound;
            }
            remoteVideo.style.display = 'block';

            const processingStatus = document.getElementById('processingStatus');
            if (processingStatus) processingStatus.style.display = 'none';
        };

        // (Opcional) ouvir DataChannel "stats" vindo do servidor (contagem de pés)
        peerConnection.ondatachannel = (event) => {
            console.log('[WebRTC] DataChannel recebido:', event.channel.label);
            const channel = event.channel;
            if (channel.label === 'stats') {
                channel.onmessage = (ev) => {
                    try {
                        const data = JSON.parse(ev.data);
                        if (data.type === 'stats' && typeof data.sort_unique_ids === 'number') {
                            updateDetectionStats(`Processando - NÚMERO DE PÉS DE SOJA: ${data.sort_unique_ids}`);
                        }
                    } catch (e) {
                        console.warn('[WebRTC] Erro ao parsear mensagem do DataChannel:', e);
                    }
                };
            }
        };

        // 3) Adiciona trilhas locais (webcam)
        stream.getTracks().forEach((t) => {
            peerConnection.addTrack(t, stream);
            console.log('[WebRTC] Track de vídeo local adicionada.');
        });

        // (Opcional) Preferir VP8
        try {
            const caps = RTCRtpSender.getCapabilities && RTCRtpSender.getCapabilities('video');
            if (caps && caps.codecs) {
                const vp8 = caps.codecs.find(c => c.mimeType === 'video/VP8');
                if (vp8) {
                    peerConnection.getTransceivers().forEach(tr => {
                        if (tr.sender && tr.setCodecPreferences) {
                            tr.setCodecPreferences([vp8, ...caps.codecs.filter(c => c !== vp8)]);
                        }
                    });
                    console.log('[WebRTC] Preferência de codec ajustada para VP8.');
                }
            }
        } catch (e) {
            console.warn('[WebRTC] Não foi possível ajustar codec preferences (OK ignorar):', e);
        }

        // 4) Observa estado da conexão
        peerConnection.onconnectionstatechange = () => {
            const state = peerConnection.connectionState;
            console.log(`[WebRTC] Estado da conexão: ${state}`);
            if (state === 'connected') {
                updateDetectionStats('Conectado - Processando...');
            } else if (state === 'failed' || state === 'closed' || state === 'disconnected') {
                updateDetectionStats('Conexão perdida');
            }
        };

        // 5) Cria a OFFER (AGORA SIM: offer existe)
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        // 6) Coleta metadados de localização
        const lat = parseFloat(document.getElementById('latitude')?.value) || null;
        const lon = parseFloat(document.getElementById('longitude')?.value) || null;
        const texto = document.getElementById('local')?.value || null;

        // 7) Envia Offer para o backend (FastAPI /webrtc/offer)
        const resp = await fetch(`${apiBase}/webrtc/offer`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${localStorage.getItem('authToken')}`
            },
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                latitude: lat,
                longitude: lon,
                local_texto: texto
            })
        });

        if (!resp.ok) {
            const err = await resp.text();
            console.error('Offer falhou:', resp.status, err);
            showNotification(`Offer falhou: ${resp.status}`, 'error');
            throw new Error(`Offer falhou: ${resp.status} - ${err}`);
        }

        const answer = await resp.json();
        await peerConnection.setRemoteDescription(answer);
        console.log('[WebRTC] Answer aplicada. Conexão estabelecida!');
        updateDetectionStats('Conectado - Processando...');

    } catch (error) {
        console.error('[WebRTC] Falha crítica ao conectar:', error);
        showNotification('Erro crítico ao conectar com WebRTC', 'error');
        stopCamera();
    }
}

function stopCamera() {
    console.log('[WebRTC] Parando câmera e conexões...');

    if (peerConnection) {
        try {
            peerConnection.getSenders().forEach(s => {
                try { s.track && s.track.stop(); } catch (e) {}
            });
            peerConnection.getReceivers().forEach(r => {
                try { r.track && r.track.stop(); } catch (e) {}
            });
            peerConnection.close();
        } catch (e) {
            console.error('[WebRTC] Erro ao fechar PeerConnection:', e);
        }
        peerConnection = null;
        console.log('[WebRTC] Peer Connection fechada.');
    }

    if (localStream) {
        localStream.getTracks().forEach(t => t.stop());
        localStream = null;
        console.log('[WebRTC] Tracks da câmera local paradas.');
    }

    updateUIAfterCameraStop();
    console.log('[WebRTC] Parada completada.');
}

function updateUIAfterCameraStop() {
    const video = document.getElementById('webcamVideo');
    const remoteVideo = document.getElementById('processedVideo');
    const cameraStatus = document.getElementById('cameraStatus');
    const processingStatus = document.getElementById('processingStatus');
    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');

    if (video) {
        video.srcObject = null;
        video.style.display = 'none';
    }
    if (remoteVideo) {
        remoteVideo.srcObject = null;
        remoteVideo.style.display = 'none';
    }
    if (cameraStatus) cameraStatus.style.display = 'flex';
    if (processingStatus) {
        processingStatus.innerHTML = '<i class="fas fa-power-off camera-icon"></i><p>Processamento parado</p>';
        processingStatus.style.display = 'flex';
    }
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    updateDetectionStats('Sistema parado');
}

function updateDetectionStats(message) {
    const statsElement = document.getElementById('detectionStats');
    if (statsElement) {
        const isActive = message.includes('Processando') || message.includes('Conectado');
        const color = isActive
            ? 'var(--primary)'
            : (message.includes('Erro') || message.includes('perdida')
                ? 'var(--accent)'
                : 'var(--gray)');

        let icon = 'fa-circle';
        let spinClass = '';

        if (isActive) {
            icon = 'fa-sync';
            spinClass = ' fa-spin';
        } else if (message.includes('Erro') || message.includes('perdida')) {
            icon = 'fa-exclamation-circle';
        } else {
            icon = 'fa-power-off';
        }

        if (message.includes('PÉS DE SOJA')) {
            statsElement.innerHTML = `<i class="fas fa-leaf" style="color: ${color};"></i> ${message}`;
        } else {
            statsElement.innerHTML = `<i class="fas ${icon}${spinClass}" style="color: ${color};"></i> ${message}`;
        }
    }
}

function updateUIAfterCameraStop() {
    const video = document.getElementById('webcamVideo');
    const remoteVideo = document.getElementById('processedVideo');
    const cameraStatus = document.getElementById('cameraStatus');
    const processingStatus = document.getElementById('processingStatus');
    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');

    if (video) {
        video.srcObject = null;
        video.style.display = 'none';
    }
    if (remoteVideo) {
        remoteVideo.srcObject = null;
        remoteVideo.style.display = 'none';
    }
    if (cameraStatus) cameraStatus.style.display = 'flex';
    if (processingStatus) {
        processingStatus.innerHTML = '<i class="fas fa-power-off camera-icon"></i><p>Processamento parado</p>';
        processingStatus.style.display = 'flex';
    }
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    updateDetectionStats('Sistema parado');
}

function updateDetectionStats(message) {
    const statsElement = document.getElementById('detectionStats');
    if (statsElement) {
        const isActive = message.includes('Processando') || message.includes('Conectado');
        const color = isActive
            ? 'var(--primary)'
            : (message.includes('Erro') || message.includes('perdida')
                ? 'var(--accent)'
                : 'var(--gray)');

        let icon = 'fa-circle';
        let spinClass = '';

        if (isActive) {
            icon = 'fa-sync';
            spinClass = ' fa-spin';
        } else if (message.includes('Erro') || message.includes('perdida')) {
            icon = 'fa-exclamation-circle';
        } else {
            icon = 'fa-power-off';
        }

        if (message.includes('Plantas:')) {
            statsElement.innerHTML = `<i class="fas fa-leaf" style="color: ${color};"></i> ${message}`;
        } else {
            statsElement.innerHTML = `<i class="fas ${icon}${spinClass}" style="color: ${color};"></i> ${message}`;
        }
    }
}

function showNotification(message, type) {
    const existingNotifications = document.querySelectorAll(`.notification.${type}`);
    existingNotifications.forEach(notif => notif.remove());

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `<i class="fas fa-${getNotificationIcon(type)}"></i> ${message}`;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.5s ease-out';
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 3500);
}

function getNotificationIcon(type) {
    const icons = { success: 'check-circle', error: 'exclamation-triangle', info: 'info-circle' };
    return icons[type] || 'info-circle';
}

// Inicialização da Aplicação

document.addEventListener('DOMContentLoaded', function () {
    setupPasswordToggle();
    initializeTheme();

    const themeToggle = document.getElementById('themeToggle');
    const themeToggleAuth = document.getElementById('themeToggleAuth');
    const themeToggleAuthRegister = document.getElementById('themeToggleAuthRegister');

    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
    if (themeToggleAuth) themeToggleAuth.addEventListener('click', toggleTheme);
    if (themeToggleAuthRegister) themeToggleAuthRegister.addEventListener('click', toggleTheme);

    const historySearch = document.getElementById('historySearch');
    if (historySearch) {
        historySearch.addEventListener('keyup', function (event) {
            if (event.key === 'Enter') filterHistory();
        });
    }

    const storedToken = localStorage.getItem('authToken');
    const storedUsername = localStorage.getItem('username');

    if (storedToken && storedUsername) {
        token = storedToken;
        currentUser = storedUsername;
        document.getElementById('userName').textContent = storedUsername;
        document.getElementById('userAvatar').textContent = storedUsername.charAt(0).toUpperCase();
        document.getElementById('authContainer').style.display = 'none';
        document.getElementById('mainApp').style.display = 'block';
        autofillLocation();
        loadHistory();
        loadDashboardStats();
    } else {
        document.getElementById('authContainer').style.display = 'flex';
        document.getElementById('mainApp').style.display = 'none';
    }

    const uploadArea = document.querySelector('.upload-area-ultra');
    if (uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, e => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.style.background = 'var(--glass)';
                uploadArea.style.borderColor = 'var(--primary)';
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.style.background = 'var(--glass)';
                uploadArea.style.borderColor = 'rgba(0, 212, 170, 0.3)';
            });
        });

        uploadArea.addEventListener('drop', e => {
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                handleFileSelect(files);
                document.getElementById('fileInput').files = files;
            } else {
                showNotification('Por favor, solte apenas arquivos de vídeo.', 'error');
            }
        });
    } else {
        console.error("Elemento .upload-area-ultra não encontrado.");
    }
});
