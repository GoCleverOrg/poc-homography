/**
 * Tenant-aware URL helper: extracts tenant_id from the query string,
 * exposes a global withTenant(url) function, and monkey-patches
 * window.fetch so every request automatically includes tenant_id.
 */
(function () {
    'use strict';

    var TENANT_ID = new URLSearchParams(window.location.search).get('tenant_id');
    if (!TENANT_ID) throw new Error('Missing required ?tenant_id= query parameter');

    window.TENANT_ID = TENANT_ID;

    window.withTenant = function withTenant(url) {
        var sep = url.includes('?') ? '&' : '?';
        return url + sep + 'tenant_id=' + encodeURIComponent(TENANT_ID);
    };

    var _origFetch = window.fetch;
    window.fetch = function (input, init) {
        if (typeof input === 'string') input = window.withTenant(input);
        return _origFetch.call(this, input, init);
    };
})();
