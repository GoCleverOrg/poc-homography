import createClient, { type Middleware } from 'openapi-fetch';
import type { paths } from './schema';

let _credentials: { username: string; password: string } | null = null;
let _onUnauthorized: (() => void) | null = null;

export function setClientCredentials(
  creds: { username: string; password: string } | null,
) {
  _credentials = creds;
}

export function setOnUnauthorized(cb: () => void) {
  _onUnauthorized = cb;
}

const authMiddleware: Middleware = {
  onRequest({ request }) {
    if (_credentials) {
      const encoded = btoa(
        `${_credentials.username}:${_credentials.password}`,
      );
      request.headers.set('Authorization', `Basic ${encoded}`);
    }
    return request;
  },
  onResponse({ response }) {
    if (response.status === 401 && _onUnauthorized) {
      _onUnauthorized();
    }
    return response;
  },
};

// --- Tenant middleware ---

let _tenantId: string | null = null;

export function setClientTenantId(tenantId: string | null) {
  _tenantId = tenantId;
}

const tenantMiddleware: Middleware = {
  onRequest({ request }) {
    if (_tenantId) {
      const url = new URL(request.url);
      if (!url.searchParams.has('tenant_id')) {
        url.searchParams.set('tenant_id', _tenantId);
        return new Request(url.toString(), request);
      }
    }
    return request;
  },
};

const client = createClient<paths>({ baseUrl: '' });
client.use(authMiddleware);
client.use(tenantMiddleware);

export default client;
