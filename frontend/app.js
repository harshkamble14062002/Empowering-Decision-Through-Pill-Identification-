const form = document.querySelector('#form');
const result = document.querySelector('#result');
const preview = document.querySelector('#annotated');
const submit = document.querySelector('#submit');
const preset = document.querySelector('#preset');
const question = document.querySelector('#question');

const kannadaClasses = {
  brand_name: 'ಬ್ರ್ಯಾಂಡ್ ಹೆಸರು',
  composition: 'ಸಂಯೋಜನೆ',
  manufacturer: 'ತಯಾರಕರು',
  ignore: 'ನಿರ್ಲಕ್ಷಿಸಲಾಗಿದೆ',
};

const kannadaReasons = {
  weak_brand_identity: 'ಬ್ರ್ಯಾಂಡ್ ಗುರುತು ದುರ್ಬಲವಾಗಿದೆ',
  weak_combined_match: 'ಒಟ್ಟಾರೆ ಹೊಂದಾಣಿಕೆ ದುರ್ಬಲವಾಗಿದೆ',
  ambiguous_candidates: 'ಹಲವು ಸಮಾನ ಅಭ್ಯರ್ಥಿಗಳಿವೆ',
};

preset.addEventListener('change', () => {
  if (preset.value) {
    question.value = preset.value;
    question.focus();
  }
});

function formatResult(data, kannada) {
  const generated = data.generation?.generated;
  const answerLabel = kannada
    ? (generated ? 'ಉತ್ತರ' : 'ಸ್ಥಳೀಯ ಪರ್ಯಾಯ ಉತ್ತರ')
    : (generated ? 'Answer' : 'Local fallback');
  const lines = [`${answerLabel}:`, data.answer || '', ''];

  if (data.medicine) {
    const medicine = data.medicine;
    const labels = kannada
      ? ['ಔಷಧಿ', 'ಸಂಯೋಜನೆ', 'ತಯಾರಕರು', 'ಹೊಂದಾಣಿಕೆ ಅಂಕ']
      : ['Medicine', 'Composition', 'Manufacturer', 'Match score'];
    const values = [
      medicine.medicine_name,
      medicine.composition,
      medicine.manufacturer,
      medicine.match_score,
    ];
    values.forEach((value, index) => lines.push(`${labels[index]}: ${value}`));
  } else {
    lines.push(kannada ? 'ಸ್ಥಿತಿ: ಗುರುತು ಖಚಿತವಾಗಿಲ್ಲ' : `Status: ${data.status}`);
    const reasons = (data.uncertainty_reasons || []).map(reason =>
      kannada ? (kannadaReasons[reason] || reason) : reason
    );
    lines.push(reasons.join(', '));
  }

  lines.push('', kannada ? 'ಚಿತ್ರದಿಂದ ಓದಿದ ಪಠ್ಯ ಪ್ರದೇಶಗಳು:' : 'OCR regions:');
  for (const region of data.ocr_regions || []) {
    const label = kannada
      ? (kannadaClasses[region.class_name] || region.class_name)
      : region.class_name;
    lines.push(`${label}: ${region.text}`);
  }
  return lines.join('\n');
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const kannada = /[\u0c80-\u0cff]/.test(question.value);
  submit.disabled = true;
  result.textContent = kannada ? 'ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ…' : 'Processing…';
  preview.hidden = true;

  try {
    const response = await fetch(form.dataset.apiEndpoint, {
      method: 'POST',
      body: new FormData(form),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Request failed');
    }
    result.textContent = formatResult(data, kannada);
    if (data.annotated_image) {
      preview.src = data.annotated_image;
      preview.hidden = false;
    }
  } catch (error) {
    result.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});
