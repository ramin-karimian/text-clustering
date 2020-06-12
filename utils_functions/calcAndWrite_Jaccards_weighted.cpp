
//  g++ -O5 -o calcJaccards_weighted calcAndWrite_Jaccards_weighted.cpp 

#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm> // for swap
#include <map>
#include<string>
#include<cmath>

using namespace std;

int intersection_size( const set<int> &A, const set<int> &B ) {
    // only get number of elements, don't build a new set
    // this assumes the sets are ordered, which std::sets are!
    int num = 0;
    set<int>::const_iterator As = A.begin(), Af = A.end(),
                             Bs = B.begin(), Bf = B.end();
    while ( As != Af && Bs != Bf ) {
        if ( *As < *Bs) {
            ++As;
        } else if ( *Bs < *As ) {
            ++Bs;
        } else {
            ++num;
            ++As;
            ++Bs;
        }
    }
    return num;
}



int main (int argc, char const *argv[]){
    // make sure args are present:
    if (!argv[1]){
        cerr << "ERROR: no input file specified" << endl;
        cerr << "usage:\n    " << argv[0] << " input.pairs output.jaccs" << endl;
        exit(1);
    }
    if (!argv[2]){
        cerr << "ERROR: no output file specified" << endl;
        cerr << "usage:\n    " << argv[0] << " input.pairs output.jaccs" << endl;
        exit(1);
    }


    // load edgelist into two arrays, and into an array of sets:
    ifstream inFile;
    inFile.open (argv[1]);
    if (!inFile) {
        cerr << "ERROR: unable to open input file" << endl;
        exit(1); // terminate with error
    }
    int ni,nj, max_node = -1;
    while (inFile >> ni >> nj ){ // scan edgelist once to get number of edges and nodes, for allocation
        if (ni > max_node){  max_node = ni;  }
        if (nj > max_node){  max_node = nj;  }
    }
    inFile.close();
    
    int num_nodes = max_node + 1; // assumes nodes are contiguous ints starting at zero
    set<int> *neighbors = NULL;
    neighbors = new set<int>[num_nodes]; // this stores the network


    inFile.open( argv[1] );
    int index = 0;
    double w;
    typedef  std::map< std::pair<int,int>, double> container;
    container ij2wij;
    while (inFile >> ni >> nj >> w){ // rescan edgelist to populate 
        neighbors[ni].insert(nj);
        neighbors[nj].insert(ni); // undirected
        ij2wij[ make_pair(ni,nj) ] = w/100;
        ij2wij[ make_pair(nj,ni) ] = w/100;
    }

    // std::map<std::int, std::double> n2a_sqrd;
    map<int, double> n2a_sqrd;
    set<int>::iterator j;
    int  n_j;
    for (int i=0; i < num_nodes; i++){
        double s = 0;
        for ( j=neighbors[i].begin(); j != neighbors[i].end(); j++){
            n_j = *j;
            s = s + ij2wij[ make_pair(i,n_j) ]; 
        }

        ij2wij[ make_pair(i,i) ] = double (s / neighbors[i].size());
        neighbors[i].insert(i); // neighbors[] is now INCLUSIVE!

        s = 0.0;
        for ( j=neighbors[i].begin(); j != neighbors[i].end(); j++){
            n_j = *j;
            s = s + pow( ij2wij[ make_pair(i,n_j)], 2); 
        }
        n2a_sqrd[i] = s;
    }
    inFile.close();
    // end load edgelist


    // do the gosh darn calculation, fool!
    
    std::string delimiter = "linked_community_results/";
    std::string delimiter1 = "_";
    std::string delimiter2 = ".txt";
    std::string ss = argv[2];
    
    // std::string s = ss.substr(ss.find(delimiter)+1);
    std::string s = ss.substr(ss.find(delimiter)+delimiter.size());

    std::string tt1 = s.substr(0, s.find(delimiter1));
    std::string sss = s.substr(s.find(delimiter1)+1);
    std::string tt2 = sss.substr(0,sss.find(delimiter2));
    // cout << argv[2]<<" "<<s<<" "<< tt1<<" "<< sss <<" "<< tt2<<endl;
    int start = std::stoi(tt1); 
    int end = std::stoi(tt2);

    FILE * jaccFile = fopen(argv[2],"w");
    // int n_i, n_j, keystone, len_int;
    int n_i, keystone, len_int;
    double curr_jacc;
    // set<int>::iterator i, j;
    set<int>::iterator i;
    for (int keystone=start; keystone < end; keystone++) { // loop over keystones 
        if ((keystone - start) % 5 == 0) 
            cout << keystone - start << "/" <<  end - start << "     for: " <<  start<< "-" <<  end<< endl;
        for ( i = neighbors[keystone].begin(); i != neighbors[keystone].end(); i++) { // neighbors of keystone
            n_i = *i;
            if (n_i == keystone)
                continue;
            
            for ( j = neighbors[keystone].begin(); j != neighbors[keystone].end(); j++ ) { // neighbors of keystone
                n_j = *j;
                if (n_j == keystone or n_i >= n_j)
                    continue;
                
                std::vector<int> common_data;
                set_intersection(neighbors[n_i].begin(),neighbors[n_i].end(),neighbors[n_j].begin(),neighbors[n_j].end(), std::back_inserter(common_data));
                double ai_dot_aj = 0;
                for (int x = 0 ; x < common_data.size() ; x++){
                    ai_dot_aj = ai_dot_aj + ij2wij[ make_pair(n_i,common_data[x])] * ij2wij[ make_pair(n_j,common_data[x])];
                }
                double curr_jacc = ai_dot_aj / ( n2a_sqrd[n_i] + n2a_sqrd[n_j] - ai_dot_aj);

                if (keystone < n_i && keystone < n_j){
                    fprintf( jaccFile, "%i\t%i\t%i\t%i\t%f\n", keystone, n_i, keystone, n_j, curr_jacc );
                } else if (keystone < n_i && keystone > n_j){
                    fprintf( jaccFile, "%i\t%i\t%i\t%i\t%f\n", keystone, n_i, n_j, keystone, curr_jacc );
                } else if (keystone > n_i && keystone < n_j){
                    fprintf( jaccFile, "%i\t%i\t%i\t%i\t%f\n", n_i, keystone, keystone, n_j, curr_jacc );
                } else {
                    fprintf( jaccFile, "%i\t%i\t%i\t%i\t%f\n", n_i, keystone, n_j, keystone, curr_jacc );
                }
            }
        }
    } // done loop over keystones
    fclose(jaccFile);

    delete [] neighbors; // all done, clean up memory...

    return 0;
}