from tkinter import Tk, Label, Text, Scrollbar, Entry, Button, DISABLED, NORMAL, END
from tkinter.constants import E, NORMAL, S
from chat import get_response, bot_name

BACKGROUND_GRAY = '#aab08d'
BACKGROUND_GRAY_ALT = '#727561'
BACKGROUND_COLOR = '#10151c'
FONT_COLOR = '#ededed'
FONT = 'Georgia 14'
FONT_BOLD = 'Georgia 13 bold'

class ChatApp:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def _setup_main_window(self):
        self.window.title('Chat')
        self.window.resizable(width=False, height=False)
        self.window.configure(width=500, height=600, bg=BACKGROUND_COLOR)

        head_label = Label(self.window, bg=BACKGROUND_COLOR, fg=FONT_COLOR, 
                        font=FONT_BOLD, pady=10, text="Welcome to Chatty!")
        head_label.place(relwidth=1)

        divider = Label(self.window, width=580, bg=BACKGROUND_GRAY)
        divider.place(relwidth=1, relheight=0.012, rely=0.07)

        self.text_widget = Text(self.window, width=20, height=2, bg=BACKGROUND_COLOR, 
                                            fg=FONT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relwidth=1, relheight=0.75, rely=0.071)
        self.text_widget.configure(cursor='arrow', state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        bottom_label = Label(self.window, bg=BACKGROUND_GRAY, height=100)
        bottom_label.place(relwidth=1, rely=0.8225)

        self.message_box = Entry(bottom_label, bg=BACKGROUND_GRAY_ALT, fg=FONT_COLOR, font=FONT)
        self.message_box.place(relwidth=0.74, relheight=0.06, rely=0.004, relx=0.011)
        self.message_box.focus()
        self.message_box.bind("<Return>", self._on_enter)

        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, 
                        bg=BACKGROUND_GRAY, command=lambda: self._on_enter(None))
        send_button.place(relwidth=0.22, relheight=0.06, rely=0.004, relx=0.77)

    def _on_enter(self, event):
        message = self.message_box.get()
        self._insert_message(message, 'You')

    def _insert_message(self, message, sender):
        if not message:
            return 

        self.message_box.delete(0, END)
        user_message = f'{sender}: {message}\n\n'
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, user_message)
        self.text_widget.configure(state=DISABLED)

        bot_message = f'{bot_name}: {get_response(message)}\n\n'
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, bot_message)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = ChatApp()
    app.run()

