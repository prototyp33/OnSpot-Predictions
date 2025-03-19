"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
exports.id = "_instrument_sentry_server_config_js";
exports.ids = ["_instrument_sentry_server_config_js"];
exports.modules = {

/***/ "(instrument)/./sentry.server.config.js":
/*!*********************************!*\
  !*** ./sentry.server.config.js ***!
  \*********************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var _sentry_nextjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @sentry/nextjs */ \"(instrument)/./node_modules/@sentry/nextjs/build/cjs/index.server.js\");\n/* harmony import */ var _sentry_nextjs__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_sentry_nextjs__WEBPACK_IMPORTED_MODULE_0__);\n// This file configures the initialization of Sentry on the server.\n// The config you add here will be used whenever the server handles a request.\n// https://docs.sentry.io/platforms/javascript/guides/nextjs/\n\nconst SENTRY_DSN = \"https://a1819f6d56af9ca958c23394f4d7f71e@o4508960674480128.ingest.de.sentry.io/4508960676511824\";\nconsole.log('Sentry Server Config - DSN:', SENTRY_DSN);\n_sentry_nextjs__WEBPACK_IMPORTED_MODULE_0__.init({\n    dsn: SENTRY_DSN,\n    // Adjust this value in production, or use tracesSampleRate for finer control\n    tracesSampleRate: 1.0,\n    // Setting this option to true will print useful information to the console while you're setting up Sentry.\n    debug: \"development\" === 'development',\n    // Enable this to see what's being sent to Sentry\n    beforeSend (event) {\n        console.log('Sending event to Sentry (server):', event.event_id);\n        return event;\n    }\n});\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGluc3RydW1lbnQpLy4vc2VudHJ5LnNlcnZlci5jb25maWcuanMiLCJtYXBwaW5ncyI6Ijs7O0FBQUEsbUVBQW1FO0FBQ25FLDhFQUE4RTtBQUM5RSw2REFBNkQ7QUFFcEI7QUFFekMsTUFBTUMsYUFBYTtBQUVuQkMsUUFBUUMsR0FBRyxDQUFDLCtCQUErQkY7QUFFM0NELGdEQUFXLENBQUM7SUFDVkssS0FBS0o7SUFFTCw2RUFBNkU7SUFDN0VLLGtCQUFrQjtJQUVsQiwyR0FBMkc7SUFDM0dDLE9BQU9DLGtCQUF5QjtJQUVoQyxpREFBaUQ7SUFDakRDLFlBQVdDLEtBQUs7UUFDZFIsUUFBUUMsR0FBRyxDQUFDLHFDQUFxQ08sTUFBTUMsUUFBUTtRQUMvRCxPQUFPRDtJQUNUO0FBQ0YiLCJzb3VyY2VzIjpbIi9Vc2Vycy9hZHJpYW5pcmFlZ3VpYWx2ZWFyL09uU3BvdF9QcmVkaWN0aXZlX01vZGVsL29uc3BvdC1kYXNoYm9hcmQvc2VudHJ5LnNlcnZlci5jb25maWcuanMiXSwic291cmNlc0NvbnRlbnQiOlsiLy8gVGhpcyBmaWxlIGNvbmZpZ3VyZXMgdGhlIGluaXRpYWxpemF0aW9uIG9mIFNlbnRyeSBvbiB0aGUgc2VydmVyLlxuLy8gVGhlIGNvbmZpZyB5b3UgYWRkIGhlcmUgd2lsbCBiZSB1c2VkIHdoZW5ldmVyIHRoZSBzZXJ2ZXIgaGFuZGxlcyBhIHJlcXVlc3QuXG4vLyBodHRwczovL2RvY3Muc2VudHJ5LmlvL3BsYXRmb3Jtcy9qYXZhc2NyaXB0L2d1aWRlcy9uZXh0anMvXG5cbmltcG9ydCAqIGFzIFNlbnRyeSBmcm9tIFwiQHNlbnRyeS9uZXh0anNcIjtcblxuY29uc3QgU0VOVFJZX0RTTiA9IFwiaHR0cHM6Ly9hMTgxOWY2ZDU2YWY5Y2E5NThjMjMzOTRmNGQ3ZjcxZUBvNDUwODk2MDY3NDQ4MDEyOC5pbmdlc3QuZGUuc2VudHJ5LmlvLzQ1MDg5NjA2NzY1MTE4MjRcIjtcblxuY29uc29sZS5sb2coJ1NlbnRyeSBTZXJ2ZXIgQ29uZmlnIC0gRFNOOicsIFNFTlRSWV9EU04pO1xuXG5TZW50cnkuaW5pdCh7XG4gIGRzbjogU0VOVFJZX0RTTixcbiAgXG4gIC8vIEFkanVzdCB0aGlzIHZhbHVlIGluIHByb2R1Y3Rpb24sIG9yIHVzZSB0cmFjZXNTYW1wbGVSYXRlIGZvciBmaW5lciBjb250cm9sXG4gIHRyYWNlc1NhbXBsZVJhdGU6IDEuMCxcbiAgXG4gIC8vIFNldHRpbmcgdGhpcyBvcHRpb24gdG8gdHJ1ZSB3aWxsIHByaW50IHVzZWZ1bCBpbmZvcm1hdGlvbiB0byB0aGUgY29uc29sZSB3aGlsZSB5b3UncmUgc2V0dGluZyB1cCBTZW50cnkuXG4gIGRlYnVnOiBwcm9jZXNzLmVudi5OT0RFX0VOViA9PT0gJ2RldmVsb3BtZW50JyxcblxuICAvLyBFbmFibGUgdGhpcyB0byBzZWUgd2hhdCdzIGJlaW5nIHNlbnQgdG8gU2VudHJ5XG4gIGJlZm9yZVNlbmQoZXZlbnQpIHtcbiAgICBjb25zb2xlLmxvZygnU2VuZGluZyBldmVudCB0byBTZW50cnkgKHNlcnZlcik6JywgZXZlbnQuZXZlbnRfaWQpO1xuICAgIHJldHVybiBldmVudDtcbiAgfSxcbn0pOyAiXSwibmFtZXMiOlsiU2VudHJ5IiwiU0VOVFJZX0RTTiIsImNvbnNvbGUiLCJsb2ciLCJpbml0IiwiZHNuIiwidHJhY2VzU2FtcGxlUmF0ZSIsImRlYnVnIiwicHJvY2VzcyIsImJlZm9yZVNlbmQiLCJldmVudCIsImV2ZW50X2lkIl0sImlnbm9yZUxpc3QiOltdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///(instrument)/./sentry.server.config.js\n");

/***/ })

};
;