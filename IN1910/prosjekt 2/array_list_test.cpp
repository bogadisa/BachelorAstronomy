#include<iostream>
#include<vector>

using namespace std;

class ArrayList {
    private:
        int *m_data;
        int m_capacity;

    public:
        int m_size;

        ArrayList() {
            m_size = 0;
            m_capacity = 10;
            m_data = new int[m_capacity];
        }

        ~ArrayList() {
            delete[] m_data;
        }

        ArrayList(vector<int> initial) {
            m_size = 0;
            m_capacity = 10;
            m_data = new int[m_capacity];

            for (int e: initial) {
                append(e);
            }
        }

        void resize() {
            m_capacity *= 2;
            int *tmp = new int[m_capacity];
            
            for (int i=0; i<m_size; i++) {
                tmp[i] = m_data[i];
            }
            delete[] m_data;
            m_data = tmp;
        }

        void append(int n) {
            if (m_size >= m_capacity){
                resize();
            }
            m_data[m_size] = n;
            m_size += 1;
        }

        int& get(int i) {
            if (0 <= i and i < m_size) {
                return m_data[i];
            } else {
                throw out_of_range("Index was out of range");
            }
        }

        int length(ArrayList x) {
            return m_size;
        }

        void print() {
            cout << "[";
            for (int i=0; i<m_size-1; i++) {
                cout << m_data[i];
                cout << ", ";
            }
            cout << m_data[m_size-1] << "]" << endl;
        }

        int& operator[] (int i) {
            if (0 <= i and i < m_size) {
                return m_data[i];
            } else {
                throw out_of_range("IndexError");
            }
        }

        void shrink_to_fit() {
            int n{2};
            while (n < m_size) {
                n *= 2;
            }
            m_capacity = n ;
        }

        int pop() {
            int x = m_data[0];
            int *tmp = new int [m_capacity];
            for (int i=1; i<m_size; i++) {
                tmp[i] = m_data[i];
            }
            delete[] m_data;
            m_data = tmp;
            return x;
        }

        ArrayList get_primes() {
            ArrayList primes;
            for (int i=0 ; i<m_size; i++) {
                int x = pop();
                if (x%2 == 0) {
                    primes.append(x);
                }
            }
            return primes;
        }

        bool is_prime(int x) {
            if (x%2 == 0) {
                return true;
            } else {
                return false;
            }
        }

        void insert(int val, int index) {
            if (m_size+1 > m_capacity){
                resize();
            }
            int *tmp = new int [m_capacity];
            for (int i=0; i<index; i++) {
                tmp[i] = m_data[i];
            }
            tmp[index] = val;
            for (int i=index+1 ; i<m_size; i++) {
                tmp[i] = m_data[i];
            }
            delete[] m_data;
            m_data = tmp;
        }

};




int main() {
    ArrayList example({0, 5, 10, 15});
    example.pop();
    example.print();

    return 0;
}
void push_front(int value) {
    size++;
    Node* tmp = head;
    head->next = new Node(value, nullptr, tmp);
    head = head->next;
  }

  int& operator[](int index) {
    Node* current = this->get_node(index);
    return current->value;
  }

  void remove(int index) {
    if (0 > index or index > size-1) {
      throw out_of_range("range_error");
    }
    Node* current = this->get_node(index);
    Node* next = current->next;
    Node* prev = current->previous;
    delete[] current;
    prev->next = next;
    next->previous = prev;
    size--;
  }