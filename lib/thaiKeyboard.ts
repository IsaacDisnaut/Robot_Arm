// Thai Kedmanee keyboard layout for KioskBoard
export const thaiKeyboardLayout = {
  en: {
    "default": [
      "1 2 3 4 5 6 7 8 9 0 {bksp}",
      "q w e r t y u i o p",
      "{shift} a s d f g h j k l {shift}",
      "{accept} . {shift} {space} {shift} {enter}"
    ],
    "shift": [
      "! @ # $ % ^ & * ( ) {bksp}",
      "Q W E R T Y U I O P",
      "{shift} A S D F G H J K L {shift}",
      "{accept} . {shift} {space} {shift} {enter}"
    ]
  },
  th: {
    "default": [
      "ฟ ห ก ด เ า ้ ่ ป ย {bksp}",
      "า ส ี ึ ุ ฺ ์ ํ ค ต",
      "{shift} ี ร น ง จ ข ค ม ว {shift}",
      "{accept} . {shift} {space} {shift} {enter}"
    ],
    "shift": [
      "เ แ โ ใ ไ ๅ ๆ ั ิ ี {bksp}",
      "ึ ื ุ ฺ ๏ ๎ ๏ ๏ ฒ ณ",
      "{shift} ด ต ถ ท ธ ฏ ฐ ฟ ฤ {shift}",
      "{accept} . {shift} {space} {shift} {enter}"
    ]
  }
};

export const thaiKeyboardOptions = {
  language: "en",
  theme: "light",
  display: "bottom",
  allowMobileKeyboard: true,
  keysIncludeString: [
    "{shift}", "{bksp}", "{enter}", "{accept}",
    "{shift}"
  ],
  // Custom key names and display
  specialCharacters: ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")",
                      "-", "_", "=", "+", "[", "]", "{", "}", ";", ":",
                      "'", "\"", "<", ">", ",", ".", "?", "/", "\\", "|"],
  customKeysInsertions: {
    "{bksp}": "Backspace",
    "{enter}": "Enter",
    "{shift}": "Shift",
    "{space}": "Space",
    "{accept}": "OK"
  }
};
