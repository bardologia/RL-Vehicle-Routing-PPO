export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export class Api {
  async request(method, url, body) {
    const options = { method };
    if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }

    let response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      throw new ApiError(0, "The console backend is unreachable.");
    }

    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new ApiError(response.status, `The backend answered ${response.status} with a body that is not JSON.`);
    }

    if (!response.ok) throw new ApiError(response.status, data.error || `HTTP ${response.status}`);
    return data;
  }

  get(url) {
    return this.request("GET", url);
  }

  post(url, body) {
    return this.request("POST", url, body || {});
  }

  put(url, body) {
    return this.request("PUT", url, body);
  }

  del(url) {
    return this.request("DELETE", url);
  }
}

export const api = new Api();
