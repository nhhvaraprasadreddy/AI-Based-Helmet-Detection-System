// ============================================================
// main.js – Shared UI logic
// ============================================================

// Live clock
function updateClock() {
  const el = document.getElementById('clock');
  if (el) el.textContent = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// Sidebar toggle
const toggleBtn = document.getElementById('sidebarToggle');
const sidebar   = document.getElementById('sidebar');
const mainContent = document.getElementById('main-content');

if (toggleBtn) {
  toggleBtn.addEventListener('click', () => {
    if (window.innerWidth <= 768) {
      sidebar.classList.toggle('open');
    } else {
      sidebar.classList.toggle('collapsed');
      mainContent.classList.toggle('expanded');
    }
  });
}

// Auto-dismiss alerts after 5 s
document.querySelectorAll('.alert-dismissible').forEach(el => {
  setTimeout(() => el.classList.add('d-none'), 5000);
});
