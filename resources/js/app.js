import './bootstrap';

import Alpine from 'alpinejs';

window.Alpine = Alpine;

Alpine.start();

document.querySelectorAll('[data-profile-panel]').forEach((panel) => {
    const toggle = panel.querySelector('[data-profile-toggle]');
    const body = panel.querySelector('[data-profile-body]');
    const label = panel.querySelector('[data-profile-toggle-label]');

    if (!toggle || !body) {
        return;
    }

    toggle.addEventListener('click', () => {
        const isOpen = panel.classList.toggle('is-open');

        toggle.setAttribute('aria-expanded', String(isOpen));
        body.setAttribute('aria-hidden', String(!isOpen));

        if (label) {
            label.textContent = isOpen ? 'Kapat' : 'Ac';
        }
    });
});

const revealTargets = document.querySelectorAll('main section, main form, main table, main details, .profile-panel');

if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches && 'IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) {
                return;
            }

            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
        });
    }, { threshold: 0.12 });

    revealTargets.forEach((target, index) => {
        target.classList.add('ui-reveal');
        target.style.animationDelay = `${Math.min(index * 45, 260)}ms`;
        revealObserver.observe(target);
    });
} else {
    revealTargets.forEach((target) => target.classList.add('is-visible'));
}
