// With JavaScript disabled, every comparison remains visible and linkable.
const panels = [...document.querySelectorAll('.comparison')];
const links = [...document.querySelectorAll('[data-comparison]')];

function selectComparison() {
  const requested = window.location.hash.slice(1);
  const selected = panels.find(panel => panel.id === requested)
    || panels.find(panel => !panel.hidden) || panels[0];
  if (!selected) return;
  for (const panel of panels) panel.hidden = panel !== selected;
  for (const link of links) {
    if (link.dataset.comparison === selected.id) link.setAttribute('aria-current', 'true');
    else link.removeAttribute('aria-current');
  }
}

selectComparison();
window.addEventListener('hashchange', () => {
  selectComparison();
  if (panels.some(panel => panel.id === window.location.hash.slice(1))) {
    document.querySelector('.comparison-nav').scrollIntoView({ block: 'start' });
  }
});
