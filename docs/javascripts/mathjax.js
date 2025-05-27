// docs/javascripts/mathjax.js
window.MathJax = {
  tex: {
    inlineMath: [ ["$", "$"], ["\\(", "\\)"] ],
    displayMath: [ ["$$", "$$"], ["\\[", "\\]"] ],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Keep these commented out for now to ensure broad scanning
    // ignoreHtmlClass: ".*|",
    // processHtmlClass: "arithmatex"
  },
  startup: {
    ready: () => {
      console.log('[MathJax Startup] MathJax v3 is ready (v3 - for main & TOC).');
      // Initial typesetting of the whole document when MathJax is first ready
      // This should catch the main content area initially.
      MathJax.startup.defaultReady();
      // An explicit typeset after defaultReady can sometimes be beneficial
      // if defaultReady doesn't catch everything immediately on complex pages.
      // MathJax.typesetPromise();
    }
  }
};

document$.subscribe(() => {
  console.log("[MathJax Re-render] document$ triggered. Attempting to re-typeset specific areas.");
  if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {

    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();

    // --- Identify Elements to Typeset ---
    // 1. Main Content Area
    //    Material for MkDocs usually wraps main content in an <article class="md-content__inner">
    //    or <div class="md-content" data-md-component="content">. Inspect with Eruda!
    const mainContentArea = document.querySelector('.md-content__inner article, .md-content[data-md-component="content"] .md-typeset');
                                        // Try '.md-content .md-typeset' or just '.md-content' if the above is too specific or not found

    // 2. TOC Container (from your previous successful selector)
    const tocContainer = document.querySelector('.md-sidebar--primary .md-nav__list'); // Use your verified selector

    const elementsToTypeset = [];
    if (mainContentArea) {
      console.log("[MathJax Re-render] Main content area found.", mainContentArea);
      elementsToTypeset.push(mainContentArea);
    } else {
      console.warn("[MathJax Re-render] Main content area NOT found with selector. Math in main content might not render on updates.");
    }

    if (tocContainer) {
      console.log("[MathJax Re-render] TOC container found.", tocContainer);
      elementsToTypeset.push(tocContainer);
    } else {
      console.warn("[MathJax Re-render] TOC container NOT found with selector. Math in TOC might not render on updates.");
    }

    if (elementsToTypeset.length > 0) {
      console.log("[MathJax Re-render] Typesetting specific elements:", elementsToTypeset);
      MathJax.typesetPromise(elementsToTypeset)
        .then(() => {
          console.log("[MathJax Re-render] typesetPromise on specific elements complete.");
        })
        .catch((err) => {
          console.error("[MathJax Re-render] typesetPromise on specific elements error:", err);
        });
    } else {
      // Fallback: If no specific elements found, try typesetting the whole document.
      // This might happen if selectors are wrong or on initial load before elements are fully ready.
      console.warn("[MathJax Re-render] No specific elements found to typeset. Falling back to general document typeset.");
      MathJax.typesetPromise()
        .then(() => {
          console.log("[MathJax Re-render] typesetPromise (fallback general) complete.");
        })
        .catch((err) => {
          console.error("[MathJax Re-render] typesetPromise (fallback general) error:", err);
        });
    }

  } else {
    console.warn("[MathJax Re-render] MathJax or typesetPromise not found.");
  }
});