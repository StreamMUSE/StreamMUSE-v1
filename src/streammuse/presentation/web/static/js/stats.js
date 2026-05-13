/**
 * Stats display for StreamMUSE Web UI
 */

const Stats = (function() {
    function updateStats(data) {
        if (data.tick !== undefined) {
            document.getElementById('stat-tick').textContent = data.tick;
        }
        
        if (data.bar !== undefined && data.beat !== undefined) {
            document.getElementById('stat-position').textContent = 
                `Bar ${data.bar}, Beat ${data.beat}`;
        }
        
        if (data.hit_rate !== undefined) {
            document.getElementById('stat-hit-rate').textContent = 
                `${(data.hit_rate * 100).toFixed(1)}%`;
        }
        
        if (data.avg_backup_level !== undefined) {
            document.getElementById('stat-backup-level').textContent = 
                data.avg_backup_level.toFixed(2);
        }
        
        if (data.round_trip_ms !== undefined) {
            document.getElementById('stat-round-trip').textContent = 
                `${data.round_trip_ms.toFixed(1)} ms`;
        }
        
        if (data.server_process_ms !== undefined) {
            document.getElementById('stat-server-time').textContent = 
                `${data.server_process_ms.toFixed(1)} ms`;
        }
        
        if (data.network_latency_ms !== undefined) {
            document.getElementById('stat-network').textContent = 
                `${data.network_latency_ms.toFixed(1)} ms`;
        }
    }
    
    function reset() {
        document.getElementById('stat-tick').textContent = '0';
        document.getElementById('stat-position').textContent = 'Bar 0, Beat 0';
        document.getElementById('stat-hit-rate').textContent = '--%';
        document.getElementById('stat-backup-level').textContent = '--';
        document.getElementById('stat-round-trip').textContent = '-- ms';
        document.getElementById('stat-server-time').textContent = '-- ms';
        document.getElementById('stat-network').textContent = '-- ms';
    }
    
    return {
        updateStats,
        reset
    };
})();
