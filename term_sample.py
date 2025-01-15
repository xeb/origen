import curses
import threading
import time
from queue import Queue
import datetime

class TerminalUI:
    def __init__(self):
        self.data_count = 0
        self.message_queue = Queue()
        self.running = True
        self.command_history = []
        self.history_index = 0
        self.status_win = None
        self.cli_win = None
        self.input_win = None
        self.stdscr = None

    def resize_windows(self):
        # Get new terminal dimensions
        max_y, max_x = self.stdscr.getmaxyx()
        
        # Delete old windows
        if self.status_win:
            del self.status_win
        if self.cli_win:
            del self.cli_win
        if self.input_win:
            del self.input_win
            
        # Create new windows with new dimensions
        self.status_win = curses.newwin(max_y // 4, max_x, 0, 0)
        self.status_win.box()
        
        self.cli_win = curses.newwin(max_y - (max_y // 4), max_x, max_y // 4, 0)
        self.cli_win.box()
        self.cli_win.scrollok(True)
        
        self.input_win = curses.newwin(1, max_x - 4, max_y - 2, 2)
        
        # Refresh all windows
        self.stdscr.clear()
        self.stdscr.refresh()
        self.status_win.refresh()
        self.cli_win.refresh()
        self.input_win.refresh()
        
        return max_y, max_x

    def setup_windows(self, stdscr):
        self.stdscr = stdscr
        
        # Configure curses
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.curs_set(1)  # Show cursor
        
        # Enable keypad and nodelay mode
        stdscr.keypad(1)
        stdscr.nodelay(1)  # Non-blocking input
        
        return self.resize_windows()

    def update_status(self):
        while self.running:
            if not self.status_win:
                continue
                
            try:
                # Clear status window content (not borders)
                self.status_win.clear()
                self.status_win.box()
                
                # Update status
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.status_win.addstr(1, 2, f"Data packets sent: {self.data_count}", curses.color_pair(1))
                self.status_win.addstr(2, 2, f"Last updated: {timestamp}", curses.color_pair(2))
                
                self.status_win.refresh()
            except curses.error:
                # Handle potential curses errors during resize
                pass
                
            time.sleep(1)

    def simulate_data(self):
        while self.running:
            self.data_count += 1
            time.sleep(0.5)  # Simulate data being sent every 0.5 seconds

    def process_command(self, command):
        if command.lower() == 'quit':
            self.running = False
            return "Shutting down..."
        elif command.lower() == 'clear':
            self.command_history.clear()
            return "History cleared"
        elif command.lower() == 'count':
            return f"Current count: {self.data_count}"
        elif command.lower() == 'help':
            return "Available commands: quit, clear, count, help"
        else:
            return f"Unknown command: {command}"

    def handle_resize(self):
        curses.endwin()
        self.stdscr.refresh()
        self.resize_windows()

    def main(self, stdscr):
        max_y, max_x = self.setup_windows(stdscr)
        current_input = ""
        cursor_x = 0

        # Start background threads
        status_thread = threading.Thread(target=self.update_status)
        data_thread = threading.Thread(target=self.simulate_data)
        status_thread.start()
        data_thread.start()

        while self.running:
            try:
                # Check for terminal resize
                if curses.is_term_resized(max_y, max_x):
                    max_y, max_x = self.resize_windows()

                # Show prompt
                self.input_win.clear()
                self.input_win.addstr(0, 0, f"> {current_input}")
                self.input_win.refresh()

                # Get input (non-blocking)
                try:
                    key = self.input_win.getch()
                except KeyboardInterrupt:
                    break

                if key == -1:  # No input
                    time.sleep(0.05)
                    continue

                if key == ord('\n'):  # Enter key
                    if current_input:
                        # Process command
                        response = self.process_command(current_input)
                        
                        # Add to history
                        self.command_history.append(current_input)
                        
                        # Clear input
                        current_input = ""
                        cursor_x = 0
                        
                        # Show response
                        max_cli_y, max_cli_x = self.cli_win.getmaxyx()
                        self.cli_win.scroll(1)
                        self.cli_win.addstr(max_cli_y-3, 2, response, curses.color_pair(1))
                        self.cli_win.box()
                        self.cli_win.refresh()
                        
                elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                    if cursor_x > 0:
                        current_input = current_input[:-1]
                        cursor_x -= 1
                        
                elif key == curses.KEY_UP:  # Up arrow
                    if self.command_history and self.history_index < len(self.command_history):
                        self.history_index += 1
                        current_input = self.command_history[-self.history_index]
                        cursor_x = len(current_input)
                        
                elif key == curses.KEY_DOWN:  # Down arrow
                    if self.history_index > 0:
                        self.history_index -= 1
                        if self.history_index == 0:
                            current_input = ""
                        else:
                            current_input = self.command_history[-self.history_index]
                        cursor_x = len(current_input)
                        
                elif key == ord('\t'):  # Tab key (ignore for now)
                    pass
                    
                elif 32 <= key <= 126:  # Printable characters
                    current_input += chr(key)
                    cursor_x += 1

            except curses.error:
                # Handle potential curses errors during resize
                self.handle_resize()

        # Cleanup
        self.running = False
        status_thread.join()
        data_thread.join()

if __name__ == "__main__":
    ui = TerminalUI()
    curses.wrapper(ui.main)