import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk
from typing import Optional, Callable, Tuple, List


class ImageVisualizer:
    """ ImageVisualizer class based on tkinter

    :param generator: Image generator
    :param nrows: Number of rows
    :param ncols: Number of columns
    :param wscreen: Screen width in pixels
    :param hscreen: Screen height in pixels
    :param focus: Callback associated with mouse click on image
    :param use_col_textboxes: Use column textboxes
    :param use_row_textboxes: Use row textboxes
    :param verbose: Verbose mode
    """

    def __init__(self,
                 generator: Callable,
                 nrows: Optional[int] = 1,
                 ncols: Optional[int] = 1,
                 wscreen: Optional[int] = 1500,
                 hscreen: Optional[int] = 900,
                 focus: Optional[Callable] = None,
                 use_col_textboxes: Optional[bool] = False,
                 use_row_textboxes: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                 ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.generator = generator
        self.use_row_textboxes = use_row_textboxes
        self.use_col_textboxes = use_col_textboxes
        self.col_textboxes_content = []
        self.row_textboxes_content = []
        self.verbose = verbose

        # Number of calls to next() function
        self.ncalls = 0

        # Compute optimal image size
        self.isize = min(int(wscreen / ncols), int(hscreen / nrows))

        # Init window
        self.win = tk.Tk()

        # Next/quit button
        self.btn_next = tk.Button(self.win, text='Next', command=self.next)
        self.btn_next.grid(row=0, column=int(ncols / 2))

        # Row and column offsets
        col_offset = 1 if use_row_textboxes else 0
        row_offset = 1 if use_col_textboxes else 0

        # Init editable text boxes
        self.col_textboxes = []
        if use_col_textboxes:
            for c in range(ncols):
                textbox = tk.Text(self.win, width=20, height=1)
                textbox.insert(tk.END, f'Col {c}')
                textbox.grid(row=1, column=c + col_offset)
                self.col_textboxes.append(textbox)
        self.row_textboxes = []
        if use_row_textboxes:
            for r in range(nrows):
                textbox = tk.Text(self.win, width=20, height=1)
                textbox.insert(tk.END, f'Row {r}')
                textbox.grid(row=r + 1 + row_offset, column=0)
                self.row_textboxes.append(textbox)

        # Init grid of images
        self.img_labels = []
        for i in range(nrows * ncols):
            r, c = int(i / ncols), int(i % ncols)
            img_label = tk.Label(self.win)
            img_label.grid(row=r + 1 + row_offset, column=c + col_offset)
            if focus is not None:
                # Attach focus callback
                img_label.bind(f'<Button-1>', focus(r, c))
            self.img_labels.append(img_label)

        # Show first images
        self.next()

        # Attach callback handling window closing
        self.win.protocol("WM_DELETE_WINDOW", self.destroy)

        # Start main loop
        self.win.mainloop()

    def next(self):
        """ Update function """
        try:
            imgs = next(self.generator)
        except StopIteration:
            # "Next" button becomes "Quit" button
            self.btn_next['text'] = 'Quit'
            self.btn_next['command'] = self.destroy
            return

        if self.verbose:
            print(f"# calls: {self.ncalls}")
        self.ncalls += 1

        # Clean display
        for label in self.img_labels:
            label.img = None
            label['image'] = None

        for img, label in zip(imgs, self.img_labels):
            if img is None:
                continue
            img = img.resize((self.isize, self.isize))
            img = ImageTk.PhotoImage(img)
            label.img = img
            label['image'] = img

    def destroy(self):
        """ Clean handling of window destruction """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Update textboxes if necessary
            self.get_textboxes()
            self.win.destroy()
            self.win = None

    def get_textboxes(self) -> Tuple[List, List]:
        """ Return current content of optional textboxes """
        if self.win is not None:
            self.col_textboxes_content = []
            self.row_textboxes_content = []
            if self.use_col_textboxes:
                self.col_textboxes_content = [txt.get("1.0", "end-1c")
                                              for txt in self.col_textboxes]
            if self.use_row_textboxes:
                self.row_textboxes_content = [txt.get("1.0", "end-1c")
                                              for txt in self.row_textboxes]
        return self.col_textboxes_content, self.row_textboxes_content
