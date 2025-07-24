"""
Port management utilities for Company Knowledge Worker
"""

import os
import socket
import subprocess
import time
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class PortManager:
    """Manages port availability and cleanup"""
    
    def __init__(self, preferred_port: int = 7860):
        self.preferred_port = preferred_port
        
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                return result == 0
        except Exception as e:
            logger.debug(f"Error checking port {port}: {e}")
            return False
    
    def get_process_using_port(self, port: int) -> Optional[dict]:
        """Get information about the process using a port"""
        try:
            # Use lsof to find process using the port
            result = subprocess.run(
                ['lsof', '-i', f':{port}', '-t'], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                if pids:
                    pid = pids[0]
                    
                    # Get process details
                    ps_result = subprocess.run(
                        ['ps', '-p', pid, '-o', 'pid,ppid,command'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if ps_result.returncode == 0:
                        lines = ps_result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            parts = lines[1].split(None, 2)
                            if len(parts) >= 3:
                                return {
                                    'pid': int(parts[0]),
                                    'ppid': int(parts[1]),
                                    'command': parts[2]
                                }
            return None
            
        except Exception as e:
            logger.debug(f"Error getting process info for port {port}: {e}")
            return None
    
    def kill_process_on_port(self, port: int, force: bool = False) -> bool:
        """Kill the process using a specific port"""
        try:
            process_info = self.get_process_using_port(port)
            if not process_info:
                logger.info(f"No process found using port {port}")
                return True
            
            pid = process_info['pid']
            command = process_info['command']
            
            logger.info(f"Found process {pid} using port {port}: {command}")
            
            # Check if it's our own process (python gradio/knowledge worker)
            if any(keyword in command.lower() for keyword in ['python', 'gradio', 'knowledge', 'app.py']):
                logger.info(f"Killing our own process {pid} on port {port}")
                
                # Try graceful termination first
                try:
                    subprocess.run(['kill', '-TERM', str(pid)], timeout=5)
                    time.sleep(2)
                    
                    # Check if process is still running
                    if self.is_port_in_use(port):
                        logger.info(f"Process {pid} still running, using SIGKILL")
                        subprocess.run(['kill', '-KILL', str(pid)], timeout=5)
                        time.sleep(1)
                    
                    return not self.is_port_in_use(port)
                    
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout killing process {pid}")
                    return False
                except Exception as e:
                    logger.error(f"Error killing process {pid}: {e}")
                    return False
            else:
                if force:
                    logger.warning(f"Force killing non-Python process {pid}: {command}")
                    try:
                        subprocess.run(['kill', '-KILL', str(pid)], timeout=5)
                        time.sleep(1)
                        return not self.is_port_in_use(port)
                    except Exception as e:
                        logger.error(f"Error force killing process {pid}: {e}")
                        return False
                else:
                    logger.warning(f"Process {pid} on port {port} is not a Python/Gradio process: {command}")
                    logger.warning(f"Use force=True to kill it anyway")
                    return False
                    
        except Exception as e:
            logger.error(f"Error killing process on port {port}: {e}")
            return False
    
    def find_available_port(self, start_port: int = None, max_attempts: int = 10) -> Optional[int]:
        """Find an available port starting from start_port"""
        if start_port is None:
            start_port = self.preferred_port
            
        for i in range(max_attempts):
            port = start_port + i
            if not self.is_port_in_use(port):
                logger.info(f"Found available port: {port}")
                return port
            
        logger.error(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")
        return None
    
    def ensure_port_available(self, port: int = None, kill_if_needed: bool = True, force_kill: bool = False) -> Optional[int]:
        """Ensure a port is available, killing processes if necessary"""
        if port is None:
            port = self.preferred_port
            
        logger.info(f"Ensuring port {port} is available...")
        
        if not self.is_port_in_use(port):
            logger.info(f"Port {port} is already available")
            return port
        
        if kill_if_needed:
            logger.info(f"Port {port} is in use, attempting to free it...")
            if self.kill_process_on_port(port, force=force_kill):
                # Wait a moment for the port to be released
                time.sleep(1)
                if not self.is_port_in_use(port):
                    logger.info(f"Successfully freed port {port}")
                    return port
                else:
                    logger.warning(f"Port {port} still in use after killing process")
            else:
                logger.warning(f"Could not kill process on port {port}")
        
        # If we can't free the preferred port, find an alternative
        logger.info(f"Looking for alternative port...")
        alternative_port = self.find_available_port(port + 1)
        if alternative_port:
            logger.info(f"Using alternative port {alternative_port}")
            return alternative_port
        
        logger.error(f"Could not ensure any port is available")
        return None
    
    def get_port_status_report(self, ports: List[int] = None) -> dict:
        """Get a status report for specified ports"""
        if ports is None:
            ports = [7860, 7861, 7862, 7863, 7864, 7865]
        
        report = {}
        for port in ports:
            is_used = self.is_port_in_use(port)
            process_info = self.get_process_using_port(port) if is_used else None
            
            report[port] = {
                'in_use': is_used,
                'process': process_info
            }
            
        return report
    
    def cleanup_old_processes(self, max_age_minutes: int = 30) -> int:
        """Clean up old Python/Gradio processes that might be hanging"""
        try:
            # Find Python processes with gradio or our app keywords
            result = subprocess.run([
                'ps', 'aux'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return 0
            
            killed_count = 0
            lines = result.stdout.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['gradio', 'knowledge_worker', 'company_knowledge']):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            # Don't kill our own process
                            if pid != os.getpid():
                                logger.info(f"Cleaning up old process {pid}: {' '.join(parts[10:])}")
                                subprocess.run(['kill', '-TERM', str(pid)], timeout=5)
                                killed_count += 1
                        except (ValueError, subprocess.TimeoutExpired):
                            continue
            
            if killed_count > 0:
                time.sleep(2)  # Give processes time to exit
                
            return killed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old processes: {e}")
            return 0