const API_BASE_URL = 'http://localhost:8000/api/auth';

interface RegisterData {
  email: string;
  password: string;
  full_name: string;
}

interface LoginData {
  email: string;
  password: string;
}

interface UserResponse {
  email: string;
  full_name: string;
  created_at: string;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
}

export class AuthService {
  private static TOKEN_KEY = 'auth_token';

  async register(data: RegisterData): Promise<UserResponse> {
    const response = await fetch(`${API_BASE_URL}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Registration failed');
    }

    return response.json();
  }

  async login(data: LoginData): Promise<TokenResponse> {
    const formData = new URLSearchParams();
    formData.append('username', data.email);
    formData.append('password', data.password);

    const response = await fetch(`${API_BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const tokenData = await response.json();
    this.setToken(tokenData.access_token);
    return tokenData;
  }

  async getCurrentUser(): Promise<UserResponse> {
    const token = this.getToken();
    if (!token) {
      throw new Error('No token found');
    }

    const response = await fetch(`${API_BASE_URL}/me`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      this.logout();
      throw new Error('Failed to get user info');
    }

    return response.json();
  }

  setToken(token: string): void {
    localStorage.setItem(AuthService.TOKEN_KEY, token);
  }

  getToken(): string | null {
    return localStorage.getItem(AuthService.TOKEN_KEY);
  }

  logout(): void {
    localStorage.removeItem(AuthService.TOKEN_KEY);
  }

  isAuthenticated(): boolean {
    return this.getToken() !== null;
  }
}

export const authService = new AuthService();