import socket

#HOST = '127.0.0.1'
PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', PORT))
sock.listen(10)
print "Start listening on port " + str(PORT)

while 1:
    #rint "listening..."
    try:
        conn, addr = sock.accept()
        pilot_msg = conn.recv(20)
        print("Accepting connection from: " + str(addr))
        print(pilot_msg)
        conn.close()

        if pilot_msg == "auth_stat":
            print "Getting person info"
            conn2, addr2 = sock.accept()
            auth_msg = conn2.recv(1024)
            #print(auth_msg)
            conn2.close()

            auth_str = auth_msg.decode('utf-8')
            with open('person_info.txt', 'w') as f:
                f.write(auth_str)


        elif pilot_msg == "photo":
            conn2, addr2 = sock.accept()
            print "Getting person photo"
            file_to_write = open('web_cap.jpg', 'wb')
            while True:
                data = conn2.recv(1024)
                    #print data
                if not data:
                    break
                #print "Writing data..."
                file_to_write.write(data)
            file_to_write.close()
            print "File received!"
            conn2.close()

    except Exception as msg:
        print str(msg)

print "Done, closing..."
sock.close()
