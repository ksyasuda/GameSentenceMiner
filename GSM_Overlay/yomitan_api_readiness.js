const YOMITAN_API_UNAVAILABLE_CODE = 'YOMITAN_API_UNAVAILABLE';
const DEFAULT_YOMITAN_API_UNAVAILABLE_ERROR = 'Yomitan extension API unavailable';

function normalizeYomitanApiError(error) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  if (typeof error === 'string' && error.trim()) {
    return error.trim();
  }
  return DEFAULT_YOMITAN_API_UNAVAILABLE_ERROR;
}

function createYomitanApiAvailabilityState() {
  return {
    ready: false,
    error: DEFAULT_YOMITAN_API_UNAVAILABLE_ERROR,
  };
}

function markYomitanApiReady() {
  return {
    ready: true,
    error: null,
  };
}

function markYomitanApiUnavailable(_state, error) {
  return {
    ready: false,
    error: normalizeYomitanApiError(error),
  };
}

function getServerVersionResponse(state, version) {
  if (state && state.ready) {
    return {
      statusCode: 200,
      payload: { version },
    };
  }
  return {
    statusCode: 503,
    payload: {
      error: normalizeYomitanApiError(state && state.error),
    },
  };
}

function ensureYomitanApiAvailable(state) {
  if (state && state.ready) {
    return;
  }
  const error = new Error(normalizeYomitanApiError(state && state.error));
  error.code = YOMITAN_API_UNAVAILABLE_CODE;
  throw error;
}

module.exports = {
  YOMITAN_API_UNAVAILABLE_CODE,
  createYomitanApiAvailabilityState,
  markYomitanApiReady,
  markYomitanApiUnavailable,
  getServerVersionResponse,
  ensureYomitanApiAvailable,
};
