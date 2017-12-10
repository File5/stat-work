class AsciiTable:

    def __init__(self, data=[], header=True, separateLines=False):
        PRECISION = 4
        
        self.data = []
        self.colomnWidth = []

        for row in data:
            newRow = []
            
            for i, item in enumerate(row):
                if type(item) is float:
                    item = round(item, PRECISION)
                    
                value = str(item)
                newRow.append(value)

                if i < len(self.colomnWidth):
                    self.colomnWidth[i] = max(len(value), self.colomnWidth[i])
                else:
                    self.colomnWidth.append(len(value))

            self.data.append(newRow)

        self.charMap = {
            'top' : '-',
            'left' : '|',
            'corner' : '+'
        }
        self.padding = 1

        self.header = header
        self.separateLines = separateLines
        
        self.highlightedRowIndex = None
        self.highlightedColIndex = None

    def highlightRow(self, index):
        self.highlightedRowIndex = index
        
    def highlightCol(self, index):
        self.highlightedColIndex = index

    def __str__(self):

        chars = self.charMap
        HIGHLIGHT_ROW = "-->"
        HIGHLIGHT_COL = "^\n|\n|"

        def _getHorizontalLine():
            result = chars['corner']

            for width in self.colomnWidth:
                result += chars['top'] * (width + 2 * self.padding) + chars['corner']

            return result

        def _getRowWithData(data):
            result = chars['left']

            for i in range(len(self.colomnWidth)):
                width = self.colomnWidth[i] + self.padding
                formatStr = '{:>' + str(width) + '}' + (' ' * self.padding)

                if i < len(data):
                    result += formatStr.format(data[i])
                else:
                    result += formatStr.format('')

                result += chars['left']

            return result

        def _getLinePrefix(i):
            highlightRowEnabled = not self.highlightedRowIndex is None
            if not highlightRowEnabled:
                return ""
            
            PREFIX = " " * len(HIGHLIGHT_ROW)

            if i == self.highlightedRowIndex:
                return HIGHLIGHT_ROW
            else:
                return PREFIX

        def _getColHighlightFooter(highlightedCol, linePrefix = ""):
            widths = [2 * self.padding + 1 + w for w in self.colomnWidth]
            widths = widths[:highlightedCol]

            markerPos = sum(widths) + 1

            markerPos += self.padding + self.colomnWidth[highlightedCol] // 2

            footer = ""
            marker = HIGHLIGHT_COL.split("\n")
            
            for char in marker:
                footer += linePrefix + " " * markerPos + char + "\n"

            return footer

        highlightColEnabled = not self.highlightedColIndex is None

        output = _getLinePrefix(-1) + _getHorizontalLine() + '\n'

        lastI = len(self.data) - 1
        for i, row in enumerate(self.data):
            output += _getLinePrefix(i) + _getRowWithData(row) + '\n'

            if (i == 0 and self.header) or (self.separateLines) or (i == lastI):
                output += _getLinePrefix(-1) + _getHorizontalLine() + '\n'

        if highlightColEnabled:
            output += _getColHighlightFooter(self.highlightedColIndex, _getLinePrefix(-1))

        return output
