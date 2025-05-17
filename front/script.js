const API_BASE = 'http://localhost:8080'; // Добавьте эту константу

async function sendData(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        credentials: 'include', // Важно для куков
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

// Обработка формы входа
document.getElementById('loginForm')?.addEventListener('submit', async function(event) {
    event.preventDefault();

    try {
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        const response = await fetch('http://localhost:8080/users/login', {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });

        if (response.status === 401) {
            const errorData = await response.json().catch(() => ({}));
            showMessage(errorData.message || 'Неверный пароль! Попробуйте еще раз', true);
            return;
        }

        if (response.status === 404) {
            const errorData = await response.json().catch(() => ({}));
            showMessage(errorData.message || 'Данный пользователь не найден, попробуйте еще раз!', true);
            return;
        }

        if (!response.ok) {
            const errorText = await response.text();
            showMessage(`Ошибка ${response.status}: ${errorText}`, true);
             return;
        }

        window.location.href = 'index.html';

    } catch (error) {
        showMessage('Ошибка сети: ' + error.message, true);
    }
});

// Функция для показа сообщений
function showMessage(text, isError = false) {
    const messageBox = document.getElementById('messageBox');
    const messageText = document.getElementById('messageText');

    messageBox.className = `message-box ${isError ? 'error' : 'success'}`;
    messageText.textContent = text;
    messageBox.classList.add('show');

    // Автоматическое скрытие через 5 секунд
    setTimeout(() => {
        messageBox.classList.remove('show');
    }, 5000);
}

// Закрытие по клику на кнопку
document.getElementById('closeMessage')?.addEventListener('click', () => {
    document.getElementById('messageBox').classList.remove('show');
});

// Обработка формы регистрации
document.getElementById('registerForm')?.addEventListener('submit', async function(event) {
    event.preventDefault();

    try {
        const email = document.getElementById('regEmail').value;
        const password = document.getElementById('regPassword').value;

        const response = await fetch('http://localhost:8080/users/register', {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email: email,
                password: password
            })
        });

        // Обработка статуса 409
        if (response.status === 409) {
            const errorData = await response.json().catch(() => ({}));
            showMessage(errorData.message || 'Пользователь с таким email уже существует', true);
            return;
        }

        const result = await response.json();

        // Обработка успешного ответа
        if (result && typeof result.success !== 'undefined') {
            if (result.success) {
                showMessage('Регистрация прошла успешно! Перенаправляем...');
                setTimeout(() => {
                    window.location.href = 'index.html';
                }, 1500);
            } else {
                showMessage(result.message || 'Ошибка регистрации', true);
            }
        } else {
            showMessage('Регистрация прошла успешно!');
            window.location.href = 'index.html';
        }
    } catch (error) {
        console.error('Registration error:', error);
        showMessage(error.message || 'Ошибка соединения с сервером', true);
    }
});

// Обработка формы загрузки файла
document.getElementById('uploadForm')?.addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('dataFile');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        window.location.href = 'model_selection.html'; // Переход на страницу выбора модели
    } else {
        alert('Ошибка загрузки файла');
    }
});

// Обработка формы выбора модели
document.getElementById('modelSelectionForm')?.addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const task = document.getElementById('task').value;

    const result = await sendData('/model/train', { task });

    if (result.success) {
        window.location.href = 'results.html'; // Переход на страницу результатов
    } else {
        alert(result.message); // Сообщение об ошибке
    }
});
