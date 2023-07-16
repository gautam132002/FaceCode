import face_recognition

def are_faces_matching(image_path_1, image_path_2):

    """
    are_faces_matching : function takes 2 image ad return true if both images have similar face
    constrain : 1 face per image

    param :  iamge_path_1 : string
    param2 : iamge_path_2 : string
    return : boolean : match = true , unmatch = false
    
    """
    try:
        
        image_1 = face_recognition.load_image_file(image_path_1)
        image_2 = face_recognition.load_image_file(image_path_2)

        
        face_encoding_1 = face_recognition.face_encodings(image_1)[0]
        face_encoding_2 = face_recognition.face_encodings(image_2)[0]

        
        result = face_recognition.compare_faces([face_encoding_1], face_encoding_2)[0]

        return result
    except IndexError:
       
        return False

#example usage 
if __name__ == "__main__":
    image_path_1 = "./13029.jpg"
    image_path_2 = "./download.jpeg"
    
    result = are_faces_matching(image_path_1, image_path_2)
    print(result)
