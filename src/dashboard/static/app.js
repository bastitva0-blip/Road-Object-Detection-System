/**
 * Road Object Detection Dashboard - Frontend Application
 */

document.addEventListener('DOMContentLoaded', function () {
    initializeControls();
    updateStats();
    updateStatus();

    setInterval(updateStats, 2000);
    setInterval(updateStatus, 5000);
});

function initializeControls() {
    const recordingToggle = document.getElementById('recording-toggle');
    const heatmapToggle = document.getElementById('heatmap-toggle');
    const confidenceSlider = document.getElementById('confidence');

    if (recordingToggle) {
        recordingToggle.addEventListener('change', async function () {
            const endpoint = this.checked ? '/api/recording/start' : '/api/recording/stop';
            try {
                const response = await fetch(endpoint, { method: 'POST' });
                const data = await response.json();
                this.checked = data.recording;
            } catch (error) {
                console.error('Recording toggle failed:', error);
                this.checked = !this.checked;
            }
        });
    }

    if (heatmapToggle) {
        heatmapToggle.addEventListener('change', async function () {
            try {
                const response = await fetch('/api/heatmap/toggle', { method: 'POST' });
                const data = await response.json();
                this.checked = data.heatmap;
            } catch (error) {
                console.error('Heatmap toggle failed:', error);
                this.checked = !this.checked;
            }
        });
    }

    if (confidenceSlider) {
        confidenceSlider.addEventListener('change', async function () {
            const value = (this.value / 100).toFixed(2);
            document.getElementById('confidence-value').textContent = value;
            try {
                await fetch('/api/config/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ confidence_threshold: parseFloat(value) }),
                });
            } catch (error) {
                console.error('Confidence update failed:', error);
            }
        });
        confidenceSlider.addEventListener('input', function () {
            document.getElementById('confidence-value').textContent = (this.value / 100).toFixed(2);
        });
    }
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        document.getElementById('uptime').textContent = `Uptime: ${Math.round(data.uptime)}s`;
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

async function updateStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        document.getElementById('vehicle-count').textContent = data.vehicle_count || 0;
        document.getElementById('pedestrian-count').textContent = data.pedestrian_count || 0;
        document.getElementById('animal-count').textContent = data.animal_count || 0;
        document.getElementById('alert-count').textContent = (data.recent_events || []).length;
        document.getElementById('fps').textContent = `FPS: ${(data.fps || 0).toFixed(1)}`;

        updateAlertsFeed(data.recent_events || []);
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

function updateAlertsFeed(events) {
    const feed = document.getElementById('alerts-feed');

    if (events.length === 0) {
        feed.innerHTML = '<div class="alert-empty">No events yet</div>';
        return;
    }

    feed.innerHTML = events.slice(0, 10).map(event => {
        const details = Object.entries(event.details || {})
            .filter(([key]) => key !== '_alert_type')
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ');
        return `<div class="alert-item">
            <strong>${event.alert_type} (${event.severity})</strong>
            <p>${details}</p>
            <small>${event.timestamp}</small>
        </div>`;
    }).join('');
}

// Export functions for testing
window.dashboardApp = {
    updateStats,
    updateStatus,
    initializeControls,
    updateAlertsFeed,
};
