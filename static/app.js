// --------------- Provider Toggle ---------------
let selectedProvider = 'gemini';

document.querySelectorAll('.provider-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.provider-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedProvider = btn.dataset.provider;

        const apiInput = document.getElementById('apiKey');
        const apiHint = document.getElementById('apiHint');

        if (selectedProvider === 'groq') {
            apiInput.placeholder = 'Enter Groq API Key...';
            apiHint.textContent = 'Using Llama 3.1 8B via Groq \u2022 Ultra-fast inference';
        } else {
            apiInput.placeholder = 'Enter Gemini API Key...';
            apiHint.textContent = 'Using Gemini 2.0 Flash \u2022 Auto-fallback to Groq if quota exhausted';
        }
    });
});

// --------------- Benchmark ---------------
document.getElementById('benchmarkBtn').addEventListener('click', async () => {
    const apiKey = document.getElementById('apiKey').value.trim();
    const promptA = document.getElementById('promptA').value.trim();
    const promptB = document.getElementById('promptB').value.trim();
    
    if (!apiKey) {
        alert("Please enter a valid API Key!");
        return;
    }
    
    if (!promptA || !promptB) {
        alert("Please fill out both Prompt A and Prompt B!");
        return;
    }

    // UI Updates
    const btn = document.getElementById('benchmarkBtn');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');
    const winnerBanner = document.getElementById('winnerBanner');
    
    btn.disabled = true;
    btn.textContent = "Benchmarking...";
    loader.style.display = "block";
    results.style.display = "none";
    winnerBanner.style.color = '';  // Reset error styling

    try {
        const response = await fetch('/api/benchmark', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                api_key: apiKey,
                prompt_a: promptA,
                prompt_b: promptB,
                provider: selectedProvider
            })
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({ error: "Server Error" }));
            throw new Error(data.error || `HTTP Error ${response.status}`);
        }
        
        const data = await response.json();

        // Show provider badge if fallback was used
        const providerBadge = document.getElementById('providerBadge');
        const providerText = document.getElementById('providerUsedText');
        if (data.provider_used) {
            const isFallback = data.provider_used.includes('fallback');
            providerText.textContent = isFallback
                ? `\u26A0\uFE0F Gemini quota hit \u2014 results powered by Groq (fallback)`
                : `Powered by ${data.provider_used.charAt(0).toUpperCase() + data.provider_used.slice(1)}`;
            providerBadge.className = `provider-badge ${isFallback ? 'fallback' : ''}`;
            providerBadge.style.display = 'block';
        } else {
            providerBadge.style.display = 'none';
        }

        // Update UI with Data
        winnerBanner.textContent = `\uD83C\uDFC6 WINNER: ${data.winner.toUpperCase()} \uD83C\uDFC6`;

        // Update Prompt A
        document.getElementById('scoreA').textContent = data.A.final_score.toFixed(2);
        document.getElementById('apiScoreA').textContent = data.A.api_quality_score + "/10";
        document.getElementById('latencyA').textContent = data.A.latency + "s";
        document.getElementById('tokensA').textContent = data.A.features.token_length;
        document.getElementById('instructionsA').textContent = data.A.features.instruction_count;
        document.getElementById('specificityA').textContent = data.A.features.specificity_score + "%";
        document.getElementById('examplesA').textContent = data.A.features.example_count;
        document.getElementById('responseA').textContent = data.A.response;

        // Update Prompt B
        document.getElementById('scoreB').textContent = data.B.final_score.toFixed(2);
        document.getElementById('apiScoreB').textContent = data.B.api_quality_score + "/10";
        document.getElementById('latencyB').textContent = data.B.latency + "s";
        document.getElementById('tokensB').textContent = data.B.features.token_length;
        document.getElementById('instructionsB').textContent = data.B.features.instruction_count;
        document.getElementById('specificityB').textContent = data.B.features.specificity_score + "%";
        document.getElementById('examplesB').textContent = data.B.features.example_count;
        document.getElementById('responseB').textContent = data.B.response;

        results.style.display = "block";

    } catch (error) {
        // Show error in-page instead of a plain alert for better UX
        const resultsDiv = document.getElementById('results');
        const providerBadge = document.getElementById('providerBadge');
        providerBadge.style.display = 'none';
        winnerBanner.textContent = error.message;
        winnerBanner.style.color = '#ff6b6b';
        resultsDiv.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = "Benchmark Prompts";
        loader.style.display = "none";
    }
});
