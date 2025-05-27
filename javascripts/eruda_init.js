// docs/javascripts/eruda_init.js
if (typeof eruda !== 'undefined') {
    eruda.init();
    console.log("Eruda initialized.");
  } else {
    console.error("Eruda script not loaded.");
  }