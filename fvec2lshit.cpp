#include <iostream>
#include <fstream>
#include <vector>
#include <boost/assert.hpp>

using namespace std;

int main (int argc, char *argv[]) {
    string input_path;
    string output_path;
    if(argc!=3)
    {
        cout << "fvec2lshkit <input> <output>" << endl;
        return 0;
    }
    input_path=argv[1];
    output_path=argv[2];
    ifstream is(input_path.c_str(), ios::binary);
    ofstream os(output_path.c_str(), ios::binary);

    int d = 4;

    os.write((char const *)&d, sizeof(d));

    is.read((char *)&d, sizeof(d));
    cout<<"dim:  "<<d<<endl;

    is.seekg(0, ios::end);

    int n = is.tellg() / (4 + d * 4);
    cout<<"n:  "<<n<<endl;
    os.write((char const *)&n, sizeof(n));
    os.write((char const *)&d, sizeof(d));

    is.seekg(0, ios::beg);

    vector<float> vec(d + 1);
    for (int i = 0; i < n; ++i) {
        is.read((char *)&vec[0], sizeof(float) * vec.size());
        if (i == 0) {
            for (int j = 0; j < d; ++j) {
                cout << vec[j+1] << ' ';
            }
            cout << endl;
        }
        BOOST_VERIFY(*(int *)&vec[0] == d);
        os.write((char const *)&vec[1], sizeof(float) * d);
    }

    os.close();
    return 0;
}
