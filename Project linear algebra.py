from PIL import Image
import numpy as np
from egcd import egcd  

#Range of letters that can be encrypted or decrypted
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#Converting either to from letter to index or from index to letter
letter_to_index = dict(zip(alphabet, range(len(alphabet))))
index_to_letter = dict(zip(range(len(alphabet)), alphabet))

#Modulus of a matrix
def matrix_mod_inv(matrix, modulus):
    det = int(np.round(np.linalg.det(matrix)))
    det_inv = egcd(det, modulus)[1] % modulus
    matrix_modulus_inv = (
        det_inv * np.round(det * np.linalg.inv(matrix)).astype(int) % modulus
    )
    return matrix_modulus_inv

#Encrypting using the   Hil Cipher method
def encrypt(message, K):
    encrypted = ""
    message_in_numbers = []

    for letter in message:
        message_in_numbers.append(letter_to_index[letter])

    split_P = [
        message_in_numbers[i : i + int(K.shape[0])]
        for i in range(0, len(message_in_numbers), int(K.shape[0]))
    ]

    for P in split_P:
        P = np.transpose(np.asarray(P))[:, np.newaxis]

        while P.shape[0] != K.shape[0]:
            P = np.append(P, letter_to_index[" "])[:, np.newaxis]

        numbers = np.dot(K, P) % len(alphabet)
        n = numbers.shape[0]

        for idx in range(n):
            number = int(numbers[idx, 0])
            encrypted += index_to_letter[number]

    return encrypted

#Decryption using the Hill cipher method
def decrypt(cipher, Kinv):
    decrypted = ""
    cipher_in_numbers = []

    for letter in cipher:
        if letter in letter_to_index:
            cipher_in_numbers.append(letter_to_index[letter])

    split_C = [
        cipher_in_numbers[i : i + int(Kinv.shape[0])]
        for i in range(0, len(cipher_in_numbers), int(Kinv.shape[0]))
    ]

    for C in split_C:
        C = np.transpose(np.asarray(C))[:, np.newaxis]
        numbers = np.dot(Kinv, C) % len(alphabet)
        n = numbers.shape[0]

        for idx in range(n):
            number = int(numbers[idx, 0])
            decrypted += index_to_letter[number]

    return decrypted


#encoding using LSB steganography
def encode_lsb(original_image_path, secret_message, encoded_image_path):
    try:
        img = Image.open(original_image_path)
    except IOError:
        print(f"Error: Unable to open the image at {original_image_path}")
        return

    if img.mode != 'RGB':
        print("Error: The image must be in RGB mode.")
        return

    pixels = list(img.getdata())

    binary_message = ''.join(format(ord(char), '08b') for char in secret_message)

    encoded_pixels = []
    binary_message_index = 0

    for pixel in pixels:
        red, green, blue = pixel

        red = (red & ~1) | int(binary_message[binary_message_index])
        binary_message_index += 1

        if binary_message_index == len(binary_message):
            break

        green = (green & ~1) | int(binary_message[binary_message_index])
        binary_message_index += 1

        if binary_message_index == len(binary_message):
            break

        blue = (blue & ~1) | int(binary_message[binary_message_index])
        binary_message_index += 1

        if binary_message_index == len(binary_message):
            break

        encoded_pixels.append((red, green, blue))

    encoded_img = Image.new(img.mode, img.size)
    encoded_img.putdata(encoded_pixels)
    
    try:
        encoded_img.save(encoded_image_path)
        print(f"Steganography complete! Encoded image saved at {encoded_image_path}")
    except IOError:
        print(f"Error: Unable to save the encoded image at {encoded_image_path}")

#decoding the image from the encoded image
def decode_lsb(encoded_image_path):

    try:
        encoded_img = Image.open(encoded_image_path)
    except IOError:
        print(f"Error: Unable to open the encoded image at {encoded_image_path}")
        return


    if encoded_img.mode != 'RGB':
        print("Error: The encoded image must be in RGB mode.")
        return

    encoded_pixels = list(encoded_img.getdata())


    binary_message = ''
    for pixel in encoded_pixels:
        red, green, blue = pixel
        binary_message += str(red & 1)
        binary_message += str(green & 1)
        binary_message += str(blue & 1)

  
    binary_message = binary_message.replace('\x00', '')

    decoded_message = ''.join(chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8))

    return decoded_message


def main():
    mode = input("Enter either 'encrypt' or 'decrypt' to either encrypt or decrypt a message: ")
    if mode == "encrypt":
        message = input("Enter the message: ").upper()
        key_matrix_input = input("Enter the key matrix (e.g., '4 5 9 11' without quotes): ")
        key_matrix_values = list(map(int, key_matrix_input.split()))
        key_matrix_size = int(np.sqrt(len(key_matrix_values)))
        K = np.matrix(key_matrix_values).reshape(key_matrix_size, key_matrix_size)
        original_image_path = r'C:\Users\HP\Desktop\python_lab\RGB_image.jpg'
        encoded_image_path = r'C:\Users\HP\Desktop\python_lab\encoded_image.png'
        encrypted_message = encrypt(message, K)
        encode_lsb(original_image_path, encrypted_message, encoded_image_path)
        print("Original message: " + message)
        print("Encrypted message: " + encrypted_message)
        img = Image.open("RGB_image.jpg")
        img1 = Image.open("encoded_image.png")
        img.show()
        img1.show()
    
    elif mode == "decrypt":
        original_image_path = r'C:\Users\HP\Desktop\python_lab\RGB_image.jpg'
        encoded_image_path = r'C:\Users\HP\Desktop\python_lab\encoded_image.png'
        decoded_message = decode_lsb(encoded_image_path)
        print(f"Decoded message: {decoded_message}")
        encrypted_message = decode_lsb(encoded_image_path).upper()
        key_matrix_input = input("Enter the key matrix (e.g., '4 5 9 11' without quotes): ")
        key_matrix_values = list(map(int, key_matrix_input.split()))
        key_matrix_size = int(np.sqrt(len(key_matrix_values)))
        K = np.matrix(key_matrix_values).reshape(key_matrix_size, key_matrix_size)
        Kinv = matrix_mod_inv(K, len(alphabet))
        decrypted_message = decrypt(encrypted_message, Kinv)
        print("Decrypted message: " + decrypted_message)
    
    else:
        print("Invalid mode. Please choose 'encrypt' or 'decrypt.'")

if __name__ == "__main__":
    main()