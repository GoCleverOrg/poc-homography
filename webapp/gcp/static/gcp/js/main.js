/*
 * GCP App JavaScript
 *
 * Note: Currently JavaScript is inlined in templates for simplicity.
 * This file is a placeholder for future JS extraction.
 */

// Utility functions that could be shared across templates
const GCPUtils = {
    /**
     * Get CSRF token from cookies for Django
     */
    getCsrfToken: function() {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, 10) === 'csrftoken=') {
                    cookieValue = decodeURIComponent(cookie.substring(10));
                    break;
                }
            }
        }
        return cookieValue;
    },

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml: function(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Format number with fixed decimals
     */
    formatNumber: function(num, decimals = 2) {
        if (typeof num !== 'number') return num;
        return num.toFixed(decimals);
    }
};
