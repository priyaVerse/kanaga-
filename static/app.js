const input = document.getElementById('imageInput');
const wrap = document.getElementById('previewWrap');
const preview = document.getElementById('preview');
const filename = document.getElementById('filename');
if (input) input.addEventListener('change', () => {
  const file = input.files && input.files[0];
  if (!file) { wrap.hidden = true; return; }
  preview.src = URL.createObjectURL(file);
  filename.textContent = file.name;
  wrap.hidden = false;
});
document.getElementById('uploadForm')?.addEventListener('submit', () => {
  const button = document.querySelector('button[type="submit"]');
  if (button) { button.disabled = true; button.textContent = 'ANALYZING…'; }
});
