* Client will upload image to server
* Server will store image in DB and tell DS/Flask to process this image
* DS will read image from DB, process image, then store processed image in DB (with metadata)
* Client will poll server for processed image and display it
