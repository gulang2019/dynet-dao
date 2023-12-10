#include <memory>
#include <stdio.h>
#include <thread>
#include <unistd.h>
#include <vector>

struct Foo {
    int val; 
    Foo(int val = 0):val(val) { printf("Foo(%d)\n", val); }
    ~Foo() { printf("~Foo(%d)\n", val); }  
};

int main() {
    std::thread t; 
    {
        std::shared_ptr<Foo> foo = std::make_shared<Foo>(1);
        std::shared_ptr<Foo> foo2 = std::make_shared<Foo>(2);
        printf("foo: %d\n", foo.get()->val);
        std::vector<std::shared_ptr<Foo>> foos;
        foos.push_back(foo);
        foos.push_back(foo2);
        printf("0 foo %d, use_count: %d\n", foo.get()->val, foo.use_count());
        printf("0 foo %d, use_count: %d\n", foo2.get()->val, foo2.use_count());
        auto func = [foos](){
            sleep(1);
            for (auto& foo: foos) {
                printf("1 foo: %d, use_count: %d\n", foo.get()->val, foo.use_count());
            }
        };
        printf("2 foo %d, use_count: %d\n", foo.get()->val, foo.use_count());
        printf("2 foo %d, use_count: %d\n", foo2.get()->val, foo2.use_count());
        t = std::thread(func);
    }
    printf("out of scope\n");
    t.join();
    printf("joined\n");
    return 0;
}