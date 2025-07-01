import cv2
import time

def test_camera():
    print("Probando diferentes configuraciones...")
    
    # Probar con DSHOW pero con configuraciones específicas
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    # Configurar propiedades de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Cámara abierta correctamente!")
    print(f"Resolución: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print("Presiona 'q' para salir")
    
    # Esperar un poco para que la cámara se inicialice
    time.sleep(2)
    
    frame_count = 0
    error_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            error_count += 1
            print(f"Error al leer frame #{frame_count + 1}, errores consecutivos: {error_count}")
            
            if error_count > 5:
                print("Demasiados errores, saliendo...")
                break
            
            time.sleep(0.1)
            continue
        
        error_count = 0  # Reset error count on successful read
        frame_count += 1
        
        # Mostrar información en el frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Presiona 'q' para salir", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar el frame
        cv2.imshow('Test Camera', frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    print(f"Test completado. Se procesaron {frame_count} frames.")

def test_different_backends():
    """Probar diferentes backends"""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for backend, name in backends:
        print(f"\n--- Probando {name} ---")
        cap = cv2.VideoCapture(0, backend)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ {name} funciona correctamente")
                cap.release()
                return backend
            else:
                print(f"✗ {name} abre pero no puede leer frames")
        else:
            print(f"✗ {name} no puede abrir la cámara")
        
        cap.release()
    
    return None

if __name__ == "__main__":
    # Primero probar qué backend funciona
    working_backend = test_different_backends()
    
    if working_backend:
        print(f"\nUsando backend que funciona...")
        test_camera()
    else:
        print("\nNingún backend funciona correctamente")