document.getElementById('benchmarkBtn').addEventListener('click', async () => {
    const apiKey = document.getElementById('apiKey').value.trim();
    const promptA = document.getElementById('promptA').value.trim();
    const promptB = document.getElementById('promptB').value.trim();
    
    if (!apiKey) {
        alert("Please enter a valid Gemini API Key!");
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
    
    btn.disabled = true;
    btn.textContent = "Benchmarking...";
    loader.style.display = "block";
    results.style.display = "none";

    try {
        const response = await fetch('/api/benchmark', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                api_key: apiKey,
                prompt_a: promptA,
                prompt_b: promptB
            })
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({ error: "Server Error (HTML returned)" }));
            throw new Error(data.error || `HTTP Error ${response.status}`);
        }
        
        const data = await response.json();

        // Update UI with Data
        document.getElementById('winnerBanner').textContent = `🏆 WINNER: ${data.winner.toUpperCase()} 🏆`;

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
        alert("Error: " + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Benchmark Prompts";
        loader.style.display = "none";
    }
});
