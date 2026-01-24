import tinyserve_ext

def main():
    a = 1
    b = 2
    c = tinyserve_ext.add(a, b)
    assert c == 3
    print("Success!")
    
main()