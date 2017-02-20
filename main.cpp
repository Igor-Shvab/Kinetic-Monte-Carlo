/*****************************************************************************
 * FILE:
 * DESCRIPTION:
	- first argument is the number of kinetic time steps
	- second argument is the number of nodes on which lammps should run (must be 1 until new developments...)
 *
 ****************************************************************************/
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
//#include "cstring"		// Lenovo Thinkpad L460
#include "string.h"		// Lenovo
#include "time.h"
//#define  tr1::arraySIZE	16000000
//#define  tr1::arraySIZE	16
#define  MASTER		0
#define M_PI 3.14159265359
#define AN 6.02214129E+23   //   1/mol
#define kB 1.3806488E-23    //   J/K
#define R 8.3144617         //   J/(mol*K)
#define hP 6.626069E-34     //   J*s
#include <vector>
#include <fstream>
#include "iostream"
//#include <tr1/tr1::array>
#include <array>
#include <string>
#include <sstream>
#include <math.h>
#include <cmath>
#include <unistd.h>
#include <iomanip>      // std::setprecision
#include <sstream>      // std::stringstream, std::stringbuf
#include <sys/types.h>
#include <dirent.h>

#include <cassert>

#include "lammps.h"     // these are LAMMPS include files
#include "input.h"
#include "atom.h"
#include "library.h"
using namespace std;
using namespace LAMMPS_NS;

#define PARALLEL

template <typename T>
std::string to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}


// DECLARE STRUCTURES (run in all processsors)

struct Atom2{
	// Atom2 types in the simulation
	std::string name;
	double mass;
};
struct MolSol{
	// molecule in solution: type, concentration, and properties.
	std::string name;
	std::vector<double> atinm;	//number of attoms of each species in the molecule
	double conc;	// number of molecules per unit volume
	double ninbox;  // at such concentration, how many such molecules are there in the simulation box?
	double Ei;	//internal energy due to Atom2 interacions, viz. lammps Denergy if the molecule is broken apart into single attoms
	double Es;	//solvation energy
	double q;	//charge.... AT SOME POINT WE WILL NEED TO CHECK ELECTRONEUTRALITY...
	//something about density of states...
};
int const MAX_n_shape_attr = 6;  // This specifies the maximum number of attributes that allow to describe the most complex particle shape in the simulation. E.g., 1 for shperical, 3 for ellipsoidal, etc..
struct PartType{
	// type of particles that can nucleate or exist initially (e.g. a substrate)
	std::string name;
	int ptid;	//NB: this tells something about the chemical nature of the particle. A more specific type for LAMMPS is then specified for each particle (see Particle structure below), in order to attribute the correct interaction. For example, if two particles of CSH type have different diameters, they can be considered for different interaction potentials in LAMMPS.
	std::vector<double> atins;	//number of attoms of each species in a molecule of solid
	double volmol;	/// molecular volume (actual only at nucleation. Further reactions can alter this value in single particles)
	double Eisp;	// internal energy per unit volume (in future, might be recomputed in every particle if they can be subjected to changes in chemistry, e.g. due to leaching or sequestration)
	double Essp;	// surface energy per unit surface (in future, it might be recomputed to account for an evolving solution chemistry, local environment, and possible chemical changes of single particles)
	double dens;	// density
	std::vector<int> RXnuc;	//list of id of possible chemical reactions for nucleation
	std::vector<int> RXgrow;	//list of id of possible chemical reactions for growth
	std::vector<int> RXdiss;	//list of id of possible chemical reactions for dissolution
	//lists for leaching and sequestration will be needed too. Solid-solid mass transfers are also left for the future..
	//something about density of states...
	std::vector<double> nuclsize;	//list of possible sizes of nuclei
	std::vector< std::array<double, 3> > nuclori;	//list of possible orientations of nuclei (here with three Euler angles, but could be converted to quaternions in future)
	std::vector<std::string> nuclshape; // list of possible shapes: names. 
	std::vector< std::array<double, MAX_n_shape_attr> > shape_attr; //list of shape attributes tr1::tr1::arrays
	std::vector<int> n_attr; // the number of significant attributes in any entry of the shape_attr vector. If entry 0 is a sphere, n_attr[0]=1, if entry 1 is an ellipsoid, n_attr[1]=3. This is usefule with MPI.
};
struct ChemRX{
	// chemical reaction: type, stoichiometry, and activation energy.
	// FOR NOW, THESE ARE ALL SOLUTION-SOLID RECTIONS. SOLID-SOLID and SOLUTION-SOLUTION REACTIONS WILL REQUIRE FURTHER IMPLEMENTATION
	std::string name;
	std::vector<double> mtosol;	//number of molecules of each species added to the solution
	std::vector<double> atosld;	//number of attoms of each species added to the solid
	double DEiso;	// activation energy of this reaction in isolated conditions
	double DVsld;	// change of volume of the solid induced by the chemical reaction
};
struct Particle{
	// the particles in the simulation
	int type;	//particle type id.. the same that will be used in lammps! Must be one of the ptid in parttype
	int typepos;	// this is computed as the actual position in the parttype vector, from 0 to its size-1
	std::vector<double> atinp;	//total number of attoms of each species in the particle
	double vol, surf, diam, mass, dens;
    int int_id; //id for the interactions in LAMMPS
	double x,y,z;
	double vx,vy,vz;
	double Ei;	// internal energy (in future, might be recomputed in every particle if they can be subjected to changes in chemistry, e.g. due to leaching or sequestration)
	double Esurf;	// surface energy (in future, it might be recomputed to account for an evolving solution chemistry, local environment, and possible chemical changes of single particles)
	double o1,o2,o3; //orientation angles. Might be converted to quaternions in future.
	std::string shape;  //e.g., Sphere, Ellipsoid, etc.. This will define the rules for the growth!
	int n_attr;		// number of relevant attributes for the shape description, i.e. relevant entries in the vector below
	double sh[MAX_n_shape_attr];	// shape descriptors
};
struct Interact{
    // the types of interactions between different particle types. These will be used to generate interaction id's
    // to be given to LAMMPS, or just to assign each particles a group id that is consistent with the LAMMPS input file and potential
    // type (or table)
    std::string name;   //e.g., Lennard-Jones , Mei , PolyMei , PairTable , Gay-Berne , etc..
    int nptypes;        // number of particle types involved in such interactions (2 for pairwise, 3 for three body, etc..)
    std::vector<int> ids;   // list of partycle type id
    int n_attr;         //the name will imply a certain number of attributes
    std::vector<double> attr; // the list of numerical values for the attributes
    /* For example:
        Lennard-Jones has two attributes
     */
};


int main (int argc, char *argv[])
{
    //check immediately that all arguments are passed:
    if (argc < 3){
        printf("ERROR: not enough input arguments from command line");
        exit(0);
    }
    clock_t start, end;
    start = clock();
    
    // DECLARE VARIABLES, tr1::arrayS, AND VECTORS (run in all processsors)
    int natsp;//atoi(argv[1]); ;  // number of Atom2ic species in solution+solids
	std::vector<Atom2> attoms;
	std::vector<MolSol> molsol;
	std::vector<ChemRX> chemRX;
	std::vector<PartType> parttype;
	std::vector<Particle> parts;
    int npart;//atoi(argv[2]); //number of particles
    double box[3];
	std::string c0fname = "conf0.init";
	std::vector< std::array<int, 5> > nucType;
    int NKtsteps =  atoi(argv[1]);  // total numnber of kinetic time steps to be performed
    int npL = atoi(argv[2]);        // number of processors that you want to use for each lammps instance (lower-level parallelization)
    //hereafter, stuff for the parallel implemenetation
	int   numtasks, taskid, rc , tag1=2, tag2=1, source;
	int *chunksize, *offset;
    MPI_Status status;
    //srand(12);
	
    double T = atof(argv[3]);
    double supersat = 0.0/* atof(argv[4])*/;
    double supersat_next = 0.0;
    
    // DECLARE FUNCTIONS (run in all procs)
	void read_chem(std::vector<Atom2> &attoms,std::vector<MolSol> &molsol, std::vector<PartType> &parttype, std::vector<ChemRX> &chemRX);
	void read_conf(std::string c0fname , double box[], std::vector<Particle> &parts);
	void read_Pchem(std::vector<Particle> &parts,std::vector<PartType> &parttype,std::vector<Atom2> &attoms);
	void print_input(std::vector<Atom2> &attoms,std::vector<MolSol> &molsol, std::vector<PartType> &parttype, std::vector<ChemRX> &chemRXdouble,double box[], std::vector<Particle> &parts);
    
    
	#ifdef PARALLEL
	// INIT MPI (run in all procs)
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);	//number of processors employed
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid); //gets the id of the current processor
    printf ("MPI task %d has started, numtasks= %d, NKtsteps= %d\n", taskid, numtasks, NKtsteps);
	#else
	taskid=0;
    numtasks=1;
	#endif
    
	bool error_input=false;	//error flag, to stop all processors if there is an error in the input
	std::vector<std::string> err_msg;	// all error messages recorded during the input phase, to be piped out at the end of it
	
    // READ INPUTS
	
	// IMPORT CHEMICAL INFO AND POSSIBLE NATURES OF NUCLEI (all processors record these)
	read_chem(attoms,molsol,parttype,chemRX);
	#ifdef PARALLEL
	MPI_Barrier(MPI_COMM_WORLD);
	#endif
	


//if (taskid==MASTER){
//srand(12);}
//else{
//srand(taskid*11);
//}

	//check that each possible chemical reaction conserves mass
	if (taskid == MASTER){
		double count_attoms[attoms.size()];
		for (int i=0; i<chemRX.size(); i++) {
			for (int ii=0; ii<attoms.size(); ii++)
	 {count_attoms[ii]=0.;
//	  std::cout<<"ChemRX["<<i<<"]= "<<chemRX[i].name<< " attoms["<<ii<<"]= "<<attoms[ii].name<<std::endl;
	 }

			for (int j=0;j<molsol.size(); j++) {
				for (int k=0;k<attoms.size(); k++) {
					count_attoms[k] += chemRX[i].mtosol[j] * molsol[j].atinm[k];
//	  std::cout<<"chemRX= "<<chemRX[i].name<<" molsol["<<j<<"]= "<<molsol[j].name<< " attoms["<<k<<"]= "<<attoms[k].name<<" mtosol["<<j<<"]= "<<chemRX[i].mtosol[j]<<" count_attoms= "<<count_attoms[k]<<std::endl;
				}
			}
			for (int j=0;j<attoms.size(); j++) {
				count_attoms[j] += chemRX[i].atosld[j];
//	  std::cout<<"ChemRX["<<i<<"]= "<<chemRX[i].name<< " attoms["<<j<<"]= "<<attoms[j].name<<" atosld= "<<chemRX[i].atosld[j]<<std::endl;
			}
			for (int ii=0; ii<attoms.size(); ii++) {
				if(count_attoms[ii]!=0.){
					printf("\n ERROR: chemical reaction %s does not respect mass balance \n",chemRX[i].name.c_str() );
					err_msg.push_back("\n ERROR: chemical reaction " + chemRX[i].name + " does not respect mass balance \n");
					error_input = true;
				}
			}
		}
	}
	
	if (taskid == MASTER){
		
		// IMPORT THE FIRST LAMMPS-STYLE CONFIGURATION
        //read_conf(c0fname,box,parts);
        
		// assign to each particle the correct id to its type position in the particle types vector
		{
			for (int i=0; i<parts.size(); i++) {
				bool found = false;
				for (int j=0; j<parttype.size(); j++) {
					if (parts[i].type == parttype[j].ptid) {
						parts[i].typepos = j;
						found = true;
					}
				}
				if (!found) {
					std::string s1 = to_string(i);
					std::string s2 = to_string(parts[i].type);
					printf("\n ERROR: particle %i has type %i which does not match with any specified particle type.",i,parts[i].type);
					err_msg.push_back("\n ERROR: particle " + s1 + " has type " + s2 + "which does not match with any specified particle type.");
					error_input = true;
				}
			}
		}

		// IMPORT CHEMICAL INFO FOR THE INITIAL CONFIGURATION (extra info compared to what lammps handles)
		//read_Pchem(parts,parttype,attoms);
        
        // READ THE INTERACTION INPUT FILE
        
        // CONSTRUCT AN INTERACTION TABLE AMONG ALL POSSIBLE PAIRS (AND NTUPLES FOR MULTI-BODY INTERACTIONS)
        
        // GIVE TO EACH PARTICLE ITS CORRECT INTERACTIO ID
        // All this part on the interactions, I leave it for later... For now lets just use a small number of simple potentials..
		
		//print a whole heap of stuff
//		print_input(attoms,molsol,parttype,chemRX,box,parts);
    }
    
	
    // INPUT ERROR HANDLING
	#ifdef PARALLEL
	// THE ERROR FLAG IS SENT FROM THE MASTER TO THE SLAVES
	if (taskid == MASTER){
		for (int dest=1; dest<numtasks; dest++) {
			MPI_Send(&error_input, 1, MPI_INT, dest, tag1, MPI_COMM_WORLD);
		}
	}
	if (taskid > MASTER) {
		source = MASTER;
		MPI_Recv(&error_input, 1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);
	}
	#endif
	// QUIT IF THERE IS AN ERROR
	if (error_input) {
		if (taskid == MASTER){
			for (int i=0; i<err_msg.size(); i++) {
				printf("%s",err_msg[i].c_str());
			}
		}
		#ifdef PARALLEL
		MPI_Finalize();
		#endif
		exit(0);
	}
    
    // THE MASTER PREPARES THE FIRST CONFIGURATION INPUT FILE FOR LAMMPS (e.g. data.init)
	if (taskid == MASTER){
        for (int i=0; i<parts.size(); i++) {
            //record each particle's position parts[i].x parts[i].y, velocity parts[i].vx parts[i].vy, and whatever needed, to the input config file for LAMMPS
            // ...to be written... for now I just use a data.init file that IGOR gave me and that I have put already into the RUN folder
        }
    }
    
    // IF NEEDED, THE MASTER PREPARES THE INTERACTION TABLE FILE FOR LAMMPS (unless someone has already prepared it consistently before, our of this code)
    if (taskid == MASTER){
    }
    
	// INIT HIGHER-LEVEL PARALLELIZATION (each processor creates its own working space)
    
    // npH SLAVE processors are used to start LAMMPS and extract+elaborate its outputs when possible
    // Here I tell each processor to create its own folder to run LAMMPS and to get a copy of the initial input configuration
    //	and input file for LAMMPS
    //
    //  NB: There is also a lower level parallelization which comes from the fact that each processor can run LAMMPS in parallel
    //		on npL cores, where npL stands for number of "SUPERSLAVE" processors used by each SLAVE processor. Overall, the number
    //		of cores used in a simulation in npH * npL
    //
    // Handling this 2-levels parallelization should not be difficult, but some issues may rise in clusters with queues.
    
    // Each Hlevel processor creates its working folder, and copies the LAMMPS executable and initial particle config there there
    std::string stskid = to_string(taskid);
    std::string workFold;		workFold = "tempLAMMPS_"+stskid;
    
    //system(("mkdir "+workFold).c_str());
    //system(("cp lmp_* "+workFold).c_str());
    //system(("cp data.init "+workFold).c_str());
    
    // if needed, copy the interaction table file too..
    std::string snpL = to_string(npL);  //converts number of lower-level processors into a string, to then use to call LAMMPS
	
	//INIT NUCLEATION (the master shares the job load)

	/*
	 // This records a list of id's that are used in the nucleation
	 // The id's refer to:
	 // - parttype
	 // - RXnuc in parttype
	 // - nuclsize in parttype
	 // - nuclshape in parttype
	 // The result is a vector of tr1::array[4]. The length of the vector is N, with:
	 //		N = SUM_i  [    parttype[i].RXnuc.size() * parttype[i].nuclsize.size() * parttype[i].nuclshape.size()   ]
	 // i.e.   (number of possible chemical reaction)   *   (number of possible nuclei sizes)  *    (number of possible nuclei shapes) 
	 //			summed over all the possible particle types
	 //
	 // N is divided in npH chunks, where npH is the number of processors involve in the high-level parallelization (see above)
	 // Each chunk is passed to a processor
	 //
	 */
		
	int Nacc[numtasks];     //an tr1::array of number of nucleations accepted during the generic iteration
	int acc_size=0.;		// total number of accepted nucleations during the generic iteration
	
	// record the id's of parttype, RXnuc in parttype, nuclsize in parttype, and nuclshape in parttype
	for (int i=0; i<parttype.size(); i++) {
	  for (int j=0; j<parttype[i].RXnuc.size(); j++) {
	    for (int k=0; k<parttype[i].nuclsize.size(); k++) {
              for (int l=0; l<parttype[i].nuclori.size(); l++) {
                for (int m=0; m<parttype[i].nuclshape.size(); m++) {
                        std::array<int, 5> temp_type;
                        temp_type[0] = i;
                        temp_type[1] = j;
                        temp_type[2] = k;
                        temp_type[3] = l;
                        temp_type[4] = m;
                        nucType.push_back(temp_type);
//std::cout<<"nucType: ptype.size_i="<<i<<" RXnuc_j="<<j<<" size_k="<<k<<" ori_l="<<l<<" shape_m="<<m<<" nucType.size="<<nucType.size()<<std::endl;
                    }
                }
            }
        }
    }
	//subdivide the nucType vector in npH chunks, such that each node gets as few jobs as possible
	chunksize = new int[numtasks];
	for (int i=0; i<numtasks; i++) {chunksize[i]=0;}
	
	{
		bool finito = false;
		chunksize[0] = nucType.size();	//first assign all nucleation jobs to the master
		while (!finito) {
			finito = true;
			// for all the slaves (i starts from 1!)
			for (int i=1; i<numtasks; i++) {
				if (chunksize[0]-chunksize[i] > 1) {
					chunksize[0]--;
					chunksize[i]++;
					finito = false;
                }
			}
		}
//for (int i=0; i<numtasks; i++) {std::cout<<"After equilibration chunksize["<<i<<"]= "<<chunksize[i]<<std::endl;}
	}

	// set offsets for each processor to read its own nucleation jobs
	offset = new int[numtasks];
	for (int i=0; i<numtasks; i++) offset[i]=0;
	
	//master's offset is 0, while for the slaves (i starts from 1) is...
	for (int i=1; i<numtasks; i++) offset[i] = offset[i-1] + chunksize[i-1];

for(int i=0; i<numtasks; i++)
std::cout<<" offset["<<i<<"]= "<<offset[i]<<" chunksize["<<i<<"]= "<<chunksize[i]<<std::endl;	
	
	// All processors use their part of the nucType vector to create their lammps input scripts for nucleation.
	// For now, I just copy the lj.in to each working folder and assign a tentative lattice size of 100
	//system(("cp in.lj "+ workFold).c_str());
	int ntry = 100;		//this should be either given consistently with LAMMPS' lattice, or read from LAMMPS output file
	//create a number of vectors that will be used later in the nucleation routine

   	std::vector<double> Utry, x_try, y_try, z_try, diam_try, o1_try, o2_try, o3_try;
	std::vector<int> ptype_try, shape_try;
	std::vector<int> ptype_acc, shape_acc;
	std::vector<double> x_acc, y_acc, z_acc, diam_acc, o1_acc, o2_acc, o3_acc;
	std::vector<int>acc; std::vector<int>acc_chunk;   // added IS

    // START MAIN LOOP
    // F&F: initiate lammps here, which will be kept running for the whole chunk
    // BEWARE!!!!! WE ARE OPENING LAMMPS IN ALL PROCESSORS, NOT ONLY ON HIGH-LEVEL PARALLELISATION ONES. THIS ASSUMES IMPLICITLY THAT THERE IS NO LOWER LEVEL PARALLELISATION, IE LAMMPS RUNS IN SERIES!

    ptype_acc.clear(); shape_acc.clear(); x_acc.clear(); y_acc.clear(); z_acc.clear(); diam_acc.clear(); o1_acc.clear(); o2_acc.clear(); o3_acc.clear(); acc.clear(); acc_chunk.clear();
    int count_acc = 0;
    
	MPI_Barrier(MPI_COMM_WORLD);
    
        LAMMPS *lmp;
        lmp = new LAMMPS(0,NULL,MPI_COMM_WORLD);
        double R_insert = 0.0, R_insert_next = 0.0, R_delete = 0.0, T_insert = 0.0, T_delete = 0.0;         // individual insert & delete rates
        int Ktstep = 0;
        int inserted = 0, deleted = 0;
        int prev_Ktstep = 0, prev_inserted = 0, prev_deleted = 0, prev_N2 = 0;
        double dt_discard = 0., KMCtime = 0., prev_ins = 0., prev_del = 0., prev_tot = 0.;
    
        double gamma = 8.8e-20;                         // Nonat homogeneous [J/nm2]        6.05e-21  - hetero
        std::vector<double> time_list;
        std::vector<double> Qrate;
        std::vector<int> Nucl_list;
        std::vector<std::string> content;
        std::ofstream outhistory, outdump, outtest, outstress;
        std::ifstream ifhistory("history.dat"), ifdump("BOX.dump"), ifbeta;
    
    if (taskid == MASTER){                              /* Setting up restart positions */
        outtest.open("test.txt", std::ios::app);
        outstress.open("stress.dat", std::ios::app);
        outstress<<"Step  Time  xxB  yyB  zzB  xyB  xzb  yzB  xxU  yyU  zzU  xyU  xzU  yzU"<<std::endl;
        std::string line;
        std::stringstream lastline;
        
        if (!ifhistory) {
            outhistory.open("history.dat");
            outhistory<<"# KMCtstep \t +N2 \t -N2 \t N2total \t Timestep \t Cumul_Time \t Cumul_R+ \t Cumul_R- \t Cumul_R"<<std::endl;
            outhistory<<"# T="<<T<<"\t beta from beta.dat"<<"\t gamma_surf="<<gamma<<" [J/nm2]"<<std::endl;
            Ktstep = 0, prev_inserted = 0, prev_deleted = 0, prev_N2 =0;
            dt_discard = 0.0, KMCtime = 0.0, prev_ins = 0.0, prev_del = 0.0, prev_tot = 0.0;
        }
        else {
            while (std::getline(ifhistory, line)) content.push_back(line);
            
            outhistory.open("history.dat", std::ios::app);
            lastline << content[content.size()-1];
            lastline >> prev_Ktstep >> prev_inserted >> prev_deleted >> prev_N2 >> dt_discard >> KMCtime >> prev_ins >> prev_del >> prev_tot;

            prev_Ktstep = prev_Ktstep + 1;
            prev_inserted = prev_inserted + 1;
            if (prev_N2 == 0) prev_N2 = 0;
            else prev_N2 = 0/*1*/;
        }
        
        if (!ifdump) {outdump.open("BOX.dump", std::ios::app);}
        else {outdump.open("BOX.dump", std::ios::app);}
    }

    /*###################################### READ BETA-TIME #######################*/
    ifbeta.open("beta_igor.dat");
    if(!ifbeta){
        std::cout<<" NO Beta-Time file "<<std::endl;
        exit(0);
    }
    else std::cout << " Reading from the beta-file" << std::endl;
    
    double nn, tt, bb;
    std::string entry, entry0;
    std::vector<double> ttime, bbeta;
    std::vector<int> nnum;
    
    getline(ifbeta, entry0);
    nnum.push_back(0.);
    ttime.push_back(0.);
    bbeta.push_back(0.);
    while (getline(ifbeta, entry))
    {
        std::stringstream sse;
        sse << entry;
        sse >> nn >> tt >> bb;
        nnum.push_back(nn);
        ttime.push_back(tt);
        bbeta.push_back(bb);
    }
    //for (int num = 0; num < nnum.size(); num++ ) std::cout << nnum[num] << "\t" << ttime[num] << "\t" << bbeta[num] << std::endl;
    ifbeta.close();
    supersat = bbeta[1];
    supersat_next = bbeta[2];
    /*###################################### END of BETA-TIME #######################*/
    
    DIR *dp;                                        /*  Choosing what data.init file to read */
    struct dirent *ep;
    dp = opendir ("./Data/");
    char src[200] , max[200]="data.init" ;
    
    if (dp != NULL)
    {
        while ((ep = readdir (dp))){
            strcpy( src , ep->d_name);
            //puts (src);
            if (strcmp(max , src) < 0) strcpy(max,src);
        }
    }
    (void) closedir (dp);
    
        time_t t;
        srand((unsigned) time(&t));
        //srand(12);
        rand();
        
        lmp->input->one("units           real");
        lmp->input->one("atom_style      sphere");
        lmp->input->one("boundary        p p f");
        lmp->input->one("pair_style      hybrid mie/cut 50 table linear 3700");
        std::cout<<"\n                   MAAAAAX FILE = "<<max<<"\n"<<std::endl;
        std::string data_init = max;
        std::stringstream ss_data;
        ss_data<<"read_data ./Data/"<<data_init;
        lmp->input->one((ss_data.str()).c_str());
        lmp->input->one("timestep        1");
        
        lmp->input->one("pair_coeff      * * table 2-2.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      1 1 mie/cut 0 100 28 14 50");
        lmp->input->one("pair_coeff      3 3 mie/cut 0 100 28 14 50");
        lmp->input->one("pair_coeff      5 5 mie/cut 0 100 28 14 50");
        lmp->input->one("pair_coeff      1 5 mie/cut 0 100 28 14 50");

        lmp->input->one("pair_coeff      1 6 table 1-6.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      2 6 table 2-6.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      3 6 table 2-6.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      5 6 table 2-6.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      6 6 table 6-6.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      6 7 table 2-2.em.dat EM28_14 185");

        lmp->input->one("pair_coeff      1 7 table 1-7.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      2 7 table 2-7.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      3 7 table 2-7.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      5 7 table 2-7.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      7 7 table 7-7.em.dat EM28_14 185");
    
        lmp->input->one("pair_coeff      1 2 table 1-2.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      1 3 table 1-2.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      2 2 table 2-2.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      2 3 table 2-2.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      2 5 table 2-2.em.dat EM28_14 185");
        lmp->input->one("pair_coeff      3 5 table 2-2.em.dat EM28_14 185");
    
        lmp->input->one("lattice         sc 100 origin 0.5 0.5 0.3 orient x  1 0 0 orient y  0 1 0 orient z  0 0 1");
        lmp->input->one("region          Box block 0 3400 0 3400 0 4500 units box");
        lmp->input->one("create_atoms    3 region Box");
        lmp->input->one("region          Layer1s block INF INF INF INF -150 -20 units box");       // STRESS-surface atoms on bottom layer
        lmp->input->one("group           Layer1s region Layer1s");
        lmp->input->one("region          Layer1b block INF INF INF INF -200 -130 units box");      // STRESS-surface atoms on bottom layer
        lmp->input->one("group           Layer1b region Layer1b");
        lmp->input->one("group           Layer1 union Layer1s Layer1b");
        lmp->input->one("group           Seed type 5");
        lmp->input->one("region          Layer2s block INF INF INF INF 6000 6130 units box");       // STRE6SS-surface atoms on upper layer
        lmp->input->one("group           Layer2s region Layer2s");
        lmp->input->one("region          Layer2b block INF INF INF INF 6130 6200 units box");       // STRESS-surface atoms on upper layer
        lmp->input->one("group           Layer2b region Layer2b");
        lmp->input->one("group           Layer2 union Layer2s Layer2b");
        lmp->input->one("group           Layer type 1");
        lmp->input->one("group           Nuc type 2");
        lmp->input->one("group           Nuc_big type 6");
        lmp->input->one("group           Nuc_small type 7");
        lmp->input->one("group           Nuclei union Nuc Nuc_big Nuc_small");
        lmp->input->one("group           Trial_Particles type 3");
        lmp->input->one("set             group all diameter 100");
        lmp->input->one("set             type 6 diameter 105");
        lmp->input->one("set             type 7 diameter 95");
        lmp->input->one("variable        n100 atom 262.24*(4.*PI*50^3/3.)/267.68");   // Vcsh=267.68[A3] Mcsh=262.24[g/mol] CSHII=2CaO SiO2 5H20 
        lmp->input->one("set             group all mass v_n100");
    
        lmp->input->one("fix             Walls all wall/reflect zlo EDGE zhi EDGE");
        lmp->input->one("fix             Freeze_1 Layer setforce 0.0 0.0 0.0");
        lmp->input->one("fix             Freeze_5 Seed setforce 0.0 0.0 0.0");                      // freeze SEED
        
        lmp->input->one("neighbor        25.0 bin");
        lmp->input->one("neigh_modify    exclude group Layer Layer");
        lmp->input->one("neigh_modify    exclude group Seed Seed");                                 // exclude SEED-SEED
        lmp->input->one("neigh_modify    exclude group Trial_Particles Trial_Particles");
        lmp->input->one("neigh_modify    every 5 delay 10 check yes page 100000 one 5000");
        
        lmp->input->one("compute         PE all pe/atom");
        lmp->input->one("compute         Radius all property/atom radius");
        lmp->input->one("compute         Temp all temp");
        lmp->input->one("compute_modify  Temp dynamic yes");
        lmp->input->one("variable        N1 equal count(Layer)");
        lmp->input->one("variable        N2 equal count(Nuclei)");
        lmp->input->one("variable        N3 equal count(Trial_Particles)");
        lmp->input->one("variable        N5 equal count(Seed)");
        lmp->input->one("variable        N6 equal count(Nuc_big)");
        lmp->input->one("variable        N7 equal count(Nuc_small)");
    
        //lmp->input->one("compute         stpa1 Nuclei stress/atom NULL virial");
        //lmp->input->one("compute         stpa2 Layer2 stress/atom NULL virial");
        //lmp->input->one("compute         stL1 Nuclei reduce sum c_stpa1[1] c_stpa1[2] c_stpa1[3] c_stpa1[4] c_stpa1[5] c_stpa1[6]");
        //lmp->input->one("compute         stL2 Layer2 reduce sum c_stpa2[1] c_stpa2[2] c_stpa2[3] c_stpa2[4] c_stpa2[5] c_stpa2[6]");

        lmp->input->one("thermo_style    custom step temp pe etotal press vol v_N2 v_N6 v_N7");
        lmp->input->one("thermo_modify   lost warn flush yes");
        lmp->input->one("thermo_modify   temp Temp");
        lmp->input->one("thermo          200");
        //lmp->input->one("dump            PE all custom 100 PE.dump id type diameter c_PE x y z");
        //lmp->input->one("dump_modify     PE sort id");
    
        for (Ktstep; Ktstep < NKtsteps; Ktstep++) {
        if (taskid == MASTER){
            std::cout<<"           ##############################################################"<<std::endl;
            std::cout<<"           ####                      Ktstep = "<<prev_Ktstep + Ktstep<<"                      ####"<<std::endl;
            std::cout<<"           ##############################################################"<<std::endl;
        }
            
            lmp->input->one("group           Nuc type 2");
            lmp->input->one("group           Nuc_big type 6");
            lmp->input->one("group           Nuc_small type 7");
            lmp->input->one("group           Nuclei union Nuc Nuc_big Nuc_small");     // indentified here so v_N2 can count new nuclei
            lmp->input->one("group           Trial_Particles type 3");
            lmp->input->one("set             group Nuc diameter 100");
            lmp->input->one("set             group Nuc_big diameter 105");
            lmp->input->one("set             group Nuc_small diameter 95");
            lmp->input->one("set             group all mass v_n100");
            
            std::stringstream ss1;
            ss1<<std::setprecision(10)<<std::fixed<<"displace_atoms  Trial_Particles random 2 2 2 "<<rand()<<" units box";
            lmp->input->one((ss1.str()).c_str());
            
            lmp->input->one("fix             Freeze_2 Nuclei setforce 0.0 0.0 0.0");   // freeze everything but Trial_Particles
            lmp->input->one("pair_coeff      2 2 mie/cut 0 100 28 14 50");             // Turn off everything but Trial_particles
            lmp->input->one("pair_coeff      1 2 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      2 5 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      3 3 mie/cut 0 100 28 14 50");
            
            lmp->input->one("pair_coeff      1 6 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      2 6 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      5 6 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      6 6 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      6 7 mie/cut 0 100 28 14 50");
            
            lmp->input->one("pair_coeff      1 7 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      2 7 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      5 7 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff      7 7 mie/cut 0 100 28 14 50");
            
            lmp->input->one("min_style       quickmin");
            lmp->input->one("min_modify      dmax 0.2 50 50 50");
            lmp->input->one("minimize        0.0 0.0 500 500");                        // Nsteps shpuld be >= sqrt(2)*half_cell
            lmp->input->one("delete_atoms    overlap 90 Trial_Particles Nuclei");
            lmp->input->one("run             0");
            
            // unsorted energies and ids of all particles
            int nlocal = static_cast<int> (lmp->atom->nlocal);
            int *aID = new int[nlocal];
            aID = ((int *) lammps_extract_atom(lmp,"id"));
            double *E = new double[nlocal];
            E = ((double *) lammps_extract_compute(lmp,"PE",1,1));                     // local E
            int *aTYPE = new int[nlocal];
            aTYPE = ((int *) lammps_extract_atom(lmp,"type"));
            double *Rad = new double[nlocal];
            Rad = ((double *) lammps_extract_compute(lmp,"Radius",1,1));
            
	//##################### Assembling unsorted aID vectors on MASTER ###########################
            int natoms = static_cast<int> (lmp->atom->natoms);
            int aIDglob[natoms];
            double Eglob[natoms];
            double Rglob[natoms];
            int nlocs[numtasks];
            
            nlocs[taskid] = nlocal;

            if (taskid > MASTER){
                int dest = MASTER;
                MPI_Send(&nlocs[taskid], 1, MPI_INT, dest, tag2, MPI_COMM_WORLD);
            }
            if (taskid == MASTER) {
                for (int i=1; i<numtasks; i++) {
                    int source = i;
                    MPI_Recv(&nlocs[i], 1, MPI_INT, source, tag2, MPI_COMM_WORLD, &status);
                }
            }
            MPI_Bcast(&nlocs, numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);

            // find the right position where the current processor should record its local ids in the global id vector
            int right_pos=0;
            for (int ii=0; ii<taskid; ii++) {right_pos += nlocs[ii];}
           
            // record local ids in global id vecotr (before communicating it to other processors)
            for (int ii=0; ii<nlocal; ii++) {
                aIDglob[right_pos+ii] = aID[ii];
                Eglob[right_pos+ii] = E[ii];
                Rglob[right_pos+ii] = Rad[ii];
            }
            
            int tag3 = 3;
            // send global id vectors to the master
            if (taskid > MASTER){
                int dest = MASTER;
                MPI_Send(&aIDglob[right_pos], nlocal, MPI_INT, dest, tag2, MPI_COMM_WORLD);
                MPI_Send(&Eglob[right_pos], nlocal, MPI_DOUBLE, dest, tag1, MPI_COMM_WORLD);
                MPI_Send(&Rglob[right_pos], nlocal, MPI_DOUBLE, dest, tag3, MPI_COMM_WORLD);
            }
            
            if (taskid == MASTER) {
                for (int i=1; i<numtasks; i++) {
                    right_pos = 0;
                    for (int ii=0; ii<i; ii++) {
                        right_pos += nlocs[ii];
                    }
                    int source = i;
                    MPI_Recv(&aIDglob[right_pos], nlocs[i], MPI_INT, source, tag2, MPI_COMM_WORLD, &status);
                    MPI_Recv(&Eglob[right_pos], nlocs[i], MPI_DOUBLE, source, tag1, MPI_COMM_WORLD, &status);
                    MPI_Recv(&Rglob[right_pos], nlocs[i], MPI_DOUBLE, source, tag3, MPI_COMM_WORLD, &status);
                }
            }   // The master now should know all the unusorted ids aIDglob
            MPI_Barrier(MPI_COMM_WORLD);
	//##################### End of Assembly #####################################################

            //sorted positions, ids, and types
            double *x = new double[3*natoms];
            lammps_gather_atoms(lmp,"x",1,3,x);
            int *aIDs= new int[natoms];
            lammps_gather_atoms(lmp,"id",0,1,aIDs);
            int *aTYPEs = new int[natoms];
            lammps_gather_atoms(lmp,"type",0,1,aTYPEs);
            
            // generate a vector of ids to go from sorted to unsorted vectors
            int *id2id = new int[natoms];     //per-atom E is in unsorted[natoms] while id & types are in sorted[3*natoms] so we must match them
            
            if (taskid == MASTER) {
                for (int ii=0; ii<natoms; ii++) {
                    for (int kk=0; kk<natoms; kk++) {
                        if (aIDglob[kk]==aIDs[ii]) {
                            id2id[ii]=kk;
                        }
                    }
                }
            }
            
            double tN1 = *((double *) lammps_extract_variable(lmp,"N1",0));
            int N1 = (int) tN1;
            double tN2 = *((double *) lammps_extract_variable(lmp,"N2",0));
            int N2 = (int) tN2;
            double tN3 = *((double *) lammps_extract_variable(lmp,"N3",0));
            int N3 = (int) tN3;
            double tN5 = *((double *) lammps_extract_variable(lmp,"N5",0));
            int N5 = (int) tN5;
            double tN6 = *((double *) lammps_extract_variable(lmp,"N6",0));
            int N6 = (int) tN6;
            double tN7 = *((double *) lammps_extract_variable(lmp,"N7",0));
            int N7 = (int) tN7;
            
            //list of rates for all trial particles and deleted nuclei
            std::vector<int> pins;      //pointing to the position of current trial particle in sorted vectors
            std::vector<double> rates;  //rates for trial particle only (consist of rate_insert and rate_delete elements)
            std::vector<double> rate_insert;  //insertion rates for trial particle only
            std::vector<double> rate_delete;  //deletion rates for trial particle only
            std::vector<double> Mtype;  //type of move associated with the rate (0 = insertion, 1 = deletion, 2 = growth)
            
            //######## INSERTION ####### WE ARE FILLING RATES VECTOR FOR EACH TRIAL PARTICLE (POSSIBLE INSERTION 3 -> 2)
            double acsh = pow(0.26768,1./3.);
            int n = ceil((4.*M_PI*pow(5.,3)/(3.*0.26768))-0.5);        // N of CSHII molecules inside particle
            double n_S = /*int (pow(n,0.667))*/ 4.*M_PI*pow(5.,2)/acsh/acsh/M_PI;  // N of CSHII molecules on surface. 2nd PI compensates for smaller a*a
            double n_R = round(5./acsh + acsh/2.);
            double omega = 4.*M_PI*pow(5.,2);                          // Surf area of particle [nm2] gamma is [J/nm2]
            double Vc = pow(100.,3);                                   // V of Cell
            double avN2 = N2/(3400.*3400.*4700./Vc);
            double phi = avN2*4.*M_PI*pow(50.,3)/(3.*Vc);
            double a3 = pow(100.,3);
            double alpha_c = -(-log(supersat)+gamma*omega/(2.*n_S*kB*T));
            double alpha_c_next = -(-log(supersat_next)+gamma*omega/(n_S*kB*T));
            double alpha_m = 1/(2.*n_S*kB*AN*T);
            double Ken = 0.0;
            double EIlow = 0.0, EIhigh = 0.0;;
            double Rad_low = 0.321963835, Rad_high = 5.;                // Size of prticle
            
            double r0 = 2.60E-4;                                         //#################### r0 - rate constant #####################
            
            if (taskid==MASTER) {
                //for all particles (in sorted vector)
                for (int ii=0; ii<natoms; ii++) {
                    if (aTYPEs[ii]==3) {
                        pins.push_back(ii);
                        Mtype.push_back(0);
                        
                        for (int ll=1; ll<=n_R; ll++) {
                            Ken = (2.*Eglob[id2id[ii]]*4184./(2.*omega*kB*AN*T) + (gamma/(1.+1./(ll)))/(2.*kB*T));
                            T_insert += exp(Ken*8.*acsh*acsh*ll/(4.*ll*ll-4.*ll+1));
                        }
                        R_insert = (r0*acsh*acsh*supersat)/T_insert;
                        
                        T_insert = 0.;
                        Ken = 0.;
                                            
                        rates.push_back(R_insert);
                        rate_insert.push_back(R_insert);
                    }
                    else if (aTYPEs[ii]==2) rate_insert.push_back(0);
                }
            }

            //######## DELETION ####### CYCLE OVER ALL THE EXISTING PARTICLES OF TYPE 2 AND TRY TO DELETE THEM TO ADD THE ASSOCIATED DELETION RATES
        if (taskid == MASTER) //std::cout<<"!!!!!!! DELETABLE NUCLEI = "<<N2<<std::endl;
            
            lmp->input->one("pair_coeff   2 3 mie/cut 0 100 28 14 50");             // Turn off trial_particles and turn on all te rest
            lmp->input->one("pair_coeff   1 2 table 1-2.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   2 2 table 2-2.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   2 5 table 2-2.em.dat EM28_14 185");
            
            lmp->input->one("pair_coeff   3 6 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff   1 6 table 1-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   2 6 table 2-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   5 6 table 2-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   6 6 table 6-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   6 7 table 2-2.em.dat EM28_14 185");
           
            lmp->input->one("pair_coeff   3 7 mie/cut 0 100 28 14 50");
            lmp->input->one("pair_coeff   1 7 table 1-7.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   2 7 table 2-7.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   5 7 table 2-7.em.dat EM28_14 185");
            lmp->input->one("pair_coeff   7 7 table 7-7.em.dat EM28_14 185");
            lmp->input->one("run	      0");
        
            double *Enew = new double[nlocal];
            Enew = ((double *) lammps_extract_compute(lmp,"PE",1,1));     // local Enew
            int *aIDnew = new int[nlocal];
            aIDnew = ((int *) lammps_extract_atom(lmp,"id"));

	//##################### Assembling NEW unsorted Enewglob vectors on MASTER ###########################
            int aIDnewglob[natoms];
            double Enewglob[natoms];
            
            nlocs[taskid] = nlocal;
  
            // find the right position where the current processor should record its local ids in the global id vector
            right_pos=0;
            for (int ii=0; ii<taskid; ii++) {right_pos += nlocs[ii];}
            
            for (int ii=0; ii<nlocal; ii++) {
                aIDnewglob[right_pos+ii] = aIDnew[ii];
                Enewglob[right_pos+ii] = Enew[ii];
            }

            if (taskid > MASTER){
                int dest = MASTER;
                MPI_Send(&aIDnewglob[right_pos], nlocal, MPI_INT, dest, tag2, MPI_COMM_WORLD);
                MPI_Send(&Enewglob[right_pos], nlocal, MPI_DOUBLE, dest, tag1, MPI_COMM_WORLD);
            }

            if (taskid == MASTER) {
                for (int i=1; i<numtasks; i++) {
                    right_pos = 0;
                    for (int ii=0; ii<i; ii++) {
                        right_pos += nlocs[ii];
                    }
                    int source = i;
                    MPI_Recv(&aIDnewglob[right_pos], nlocs[i], MPI_INT, source, tag2, MPI_COMM_WORLD, &status);
                    MPI_Recv(&Enewglob[right_pos], nlocs[i], MPI_DOUBLE, source, tag1, MPI_COMM_WORLD, &status);
                }
            }   // The master now should know NEW usorted Enewglob
            MPI_Barrier(MPI_COMM_WORLD);
	//##################### End of Assembly #####################################################
            
            int PPos = 0, pos, posit;
            double dt;
            if (taskid == MASTER){
                for (int ii=0; ii<natoms; ii++) {
                    if (aIDnewglob[ii] != aIDglob[ii]) {
                        std::cout<<"ERROR WITH THE IDS. MORE WORK NEEDED"<<std::endl;
                        exit(0);
                    }
                }

                for (int ii=0; ii<natoms; ii++) {
                    if (aTYPEs[ii]==2 || aTYPEs[ii]==6 || aTYPEs[ii]==7) {
                        pins.push_back(ii);
                        Mtype.push_back(1);
                        
                        for (int ll=1; ll<=n_R; ll++) {
                            Ken = (2.*Enewglob[id2id[ii]]*4184./(2.*omega*kB*AN*T) + (gamma/(1.+1./(ll)))/(2.*kB*T));
                            T_delete += exp(-Ken*8.*acsh*acsh*ll/(4.*ll*ll + 4.*ll + 1));
                        }
                        R_delete = (r0*acsh*acsh)/T_delete;
                                            
                        T_delete = 0.;
                        Ken = 0.;
                                            
                        rates.push_back(R_delete);
                        rate_delete.push_back(R_delete);
                    }
                    else if (aTYPEs[ii]==3) rate_delete.push_back(0);
                }
                //######### END of DELETION #################
                
                double totRATEins= 0;
                double totRATEdel= 0;
                
                //######### CUMULATIVE rates ################
                for (int ii=1; ii<rates.size(); ii++) {
                    rates[ii]+=rates[ii-1];
                    
                    if (ii==N3-1) {
                        totRATEins = rates[ii];
                    }
                    rate_insert[ii]+=rate_insert[ii-1];
                    rate_delete[ii]+=rate_delete[ii-1];
                }
                totRATEdel = rates[rates.size()-1]-totRATEins;
                
                //######### BINARY SEARCH ################# find event with chosen cumulative Rate
                double chosenR = (double)rand() / (double)RAND_MAX *rates[rates.size()-1];
                int pre = 0;
                int post=rates.size()-1;
                
                while (pre < post) {
                    pos = (int)((pre+post)/2.);
                    if (rates[pos] < chosenR) pre=pos+1;
                    else post=pos;
                }
                pos=pre;
                
                //######### BINARY SEARCH in beta_igor.dat ##### find next closest ttime
                int preT = 1;
                int postT = ttime.size()-1;
                
                while (preT < postT) {
                    posit = (int)((preT+postT)/2.);
                    if (ttime[posit] < KMCtime) preT=posit+1;
                    else postT=posit;
                }
                if(preT == 1) posit=preT+1;
                else posit=preT;
                
                std::cout<<" ########## ATTEMPTED NUCLEI Ktstep= "<<Ktstep<<" pos="<<pos<<"  atomic_id="<<aIDs[pins[pos]]<<" MType="<<Mtype[pos]<<" ratE="<<std::setprecision(10)<<rates[pos]-rates[pos-1]<<" x="<<x[3*pins[pos]]<<" y="<<x[3*pins[pos]+1]<<" z="<<x[3*pins[pos]+2]<<std::endl;
                //######### END OF SEARCH #################
                
	/*############################################ TIME stuff ##############################################*/
                supersat = bbeta[posit-1];
                supersat_next = bbeta[posit];
                Qrate.push_back(rates[rates.size()-1]);
                Nucl_list.push_back(N2);
                time_list.push_back(KMCtime);
                double totRATEnew = 0.;
                double u1 = 0.;
                
                // guess time increment if beta stays constant
                dt= 1./(rates[rates.size()-1])*log((double)RAND_MAX / (double)rand());
                
                //if dt does not take you to the next beta, just add dt to KMCtime
                if (KMCtime + dt <= ttime[posit]) {
                    KMCtime += dt;
                }
                else{
                    // if dt takes you to further betas, apply corrections until you find the right beta step that give consistent results
                    double area_left = dt * Qrate[Ktstep];
                    area_left -= (ttime[posit]-KMCtime)*Qrate[Ktstep];
                    
                    while (area_left > 0) {
                        KMCtime = ttime[posit];
                        totRATEnew = totRATEdel + totRATEins / supersat * bbeta[posit];
                        posit++;
                        area_left -= totRATEnew * (ttime[posit]-ttime[posit-1]);
                    }
                    
                    u1 = (totRATEnew * (ttime[posit]-ttime[posit-1]) + area_left) / totRATEnew;
                    KMCtime += u1;
                }

	/*############################################ END of TIME stuff ##############################################*/
                
                /* ######################## WRITING BOX and HISTORY files ############################## */
                outhistory<<prev_Ktstep + Ktstep<<"\t"<<prev_inserted + inserted<<"\t"<<prev_deleted + deleted<<"\t"<<prev_N2 + N2<<"\t"<<std::setprecision(10)<<dt<<"\t"<<KMCtime<<"\t"<<std::setprecision(10)<</*prev_ins + */rate_insert[rate_insert.size()-1]<<"\t"<</*prev_del + */rate_delete[rate_delete.size()-1]<<"\t"<</*prev_tot + */rates[rates.size()-1]<<std::endl;
 
                if((prev_Ktstep + Ktstep) % 1 == 0)
                {
                    outdump<<natoms - N3<<std::endl;
                    outdump<<"id   type   radius    x    y    z"<<std::endl;
                    for (int i = 0; i < natoms; i++)
                        if (aTYPEs[i] != 3) outdump<<aIDs[i]<<" "<<aTYPEs[i]<<" "<<Rglob[id2id[i]]<<" "<<x[3*i]<<" "<<x[3*i+1]<<" "<<x[3*i+2]<<std::endl;
                }

                PPos = pins[pos];
            }
  
            if (taskid == MASTER){          // Send-Receive positions in x-array for chosen particle
                for (int dest=1; dest<numtasks; dest++) {
                    MPI_Send(&PPos, 1, MPI_INT, dest, tag1, MPI_COMM_WORLD);
                }
            }
            if (taskid > MASTER) {
                source = MASTER;
                MPI_Recv(&PPos, 1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);
            }
            
            int Mtype_sr;                   // Mtype send-receive
            
            if (taskid == MASTER) {
                Mtype_sr = Mtype[pos];

                for (int dest=1; dest<numtasks; dest++) {
                    MPI_Send(&Mtype_sr, 1, MPI_INT, dest, tag1, MPI_COMM_WORLD);
                }
            }
            if (taskid > MASTER) {          // Send-Receive chosen particle M-type
                source = MASTER;
                MPI_Recv(&Mtype_sr, 1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);
            }
            
            double randnum = (double)rand()/(double)RAND_MAX;
            
            if (Mtype_sr==0 && randnum<=0.2) {
                //ACTUALL NUCLEI INSTERION on every processor. Due to domain decomposit apparently only one processor inserts/delets
                std::stringstream ss7;
                ss7<<std::setprecision(10)<<std::fixed<<"create_atoms 7 single "<<x[3*PPos]<<" "<<x[3*PPos+1]<<" "<<x[3*PPos+2]<<" units box";
            if (taskid == MASTER) std::cout<<" ########## randnum="<<randnum<<" ATTENTION INSERTION CHOSEN "<<ss7.str()<<std::endl;

                lmp->input->one((ss7.str()).c_str());
                lmp->input->one("set          group Nuc_small diameter 95");
                inserted += 1;
            }

            else if (Mtype_sr==0 && randnum>=0.8) {
                //ACTUALL NUCLEI INSTERION on every processor. Due to domain decomposit apparently only one processor inserts/delets
                std::stringstream ss6;
                ss6<<std::setprecision(10)<<std::fixed<<"create_atoms 6 single "<<x[3*PPos]<<" "<<x[3*PPos+1]<<" "<<x[3*PPos+2]<<" units box";
                if (taskid == MASTER) std::cout<<" ########## randnum="<<randnum<<" ATTENTION INSERTION CHOSEN "<<ss6.str()<<std::endl;
                
                lmp->input->one((ss6.str()).c_str());
                lmp->input->one("set          group Nuc_big diameter 105");
                inserted += 1;
            }
            
            else if (Mtype_sr==0 && randnum>=0.2 && randnum<=0.8) {
                //ACTUALL NUCLEI INSTERION on every processor. Due to domain decomposit apparently only one processor inserts/delets
                std::stringstream ss;
                ss<<std::setprecision(10)<<std::fixed<<"create_atoms 2 single "<<x[3*PPos]<<" "<<x[3*PPos+1]<<" "<<x[3*PPos+2]<<" units box";
                if (taskid == MASTER) std::cout<<" ########## randnum="<<randnum<<" ATTENTION INSERTION CHOSEN "<<ss.str()<<std::endl;
                
                lmp->input->one((ss.str()).c_str());
                lmp->input->one("set          group Nuc diameter 100");
                inserted += 1;
            }
            
            else if(Mtype_sr==1){
                //ACTUAL NUCLEI DELETION on every processor
                std::stringstream ss;
                ss<<"set atom "<<aIDs[PPos]<<" type 4";
            if (taskid == MASTER) std::cout<<" ########## ATTENTION DELETION CHOSEN pos="<<pos<<" atomic_id="<<aIDs[PPos]<<" "<<ss.str()<<std::endl;
            
                lmp->input->one((ss.str()).c_str());
                lmp->input->one("group        Deletable type 4");
                lmp->input->one("delete_atoms group Deletable");
                deleted += 1;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            lmp->input->one("delete_atoms     group Trial_Particles");
            lmp->input->one("unfix            Freeze_2");
            lmp->input->one("set              group all mass v_n100");
            lmp->input->one("min_style        cg");
            lmp->input->one("min_modify       dmax 0.1 50 50 50");
            lmp->input->one("minimize         0.0 0.0 500 500");
            
            std::stringstream ss_restart;
            if ((prev_Ktstep + Ktstep) <= 9) ss_restart<<"write_data ./Data/data.init.000000"<<prev_Ktstep + Ktstep<<".*";
            else if ((prev_Ktstep + Ktstep) <= 99) ss_restart<<"write_data ./Data/data.init.00000"<<prev_Ktstep + Ktstep<<".*";
            else if ((prev_Ktstep + Ktstep) <= 999) ss_restart<<"write_data ./Data/data.init.0000"<<prev_Ktstep + Ktstep<<".*";
            else if ((prev_Ktstep + Ktstep) <= 9999) ss_restart<<"write_data ./Data/data.init.000"<<prev_Ktstep + Ktstep<<".*";
            else if ((prev_Ktstep + Ktstep) <= 99999) ss_restart<<"write_data ./Data/data.init.00"<<prev_Ktstep + Ktstep<<".*";
            else if ((prev_Ktstep + Ktstep) <= 999999) ss_restart<<"write_data ./Data/data.init.0"<<prev_Ktstep + Ktstep<<".*";
            else ss_restart<<"write_data ./Data/data.init."<<prev_Ktstep + Ktstep<<".*";
            
            if((prev_Ktstep + Ktstep) % 50 == 0)
            lmp->input->one((ss_restart.str()).c_str());

            /*########### STRESS TENSOR COMPONENTS ############*/
            /*
            lmp->input->one("unfix            Freeze_1");
            lmp->input->one("pair_coeff       1 1 665 100 28 14");
            lmp->input->one("run              0");
            
            double Vsurf = 3000.*3000.*200.;
            double Vbulk = 3000.*3000.*6000.;
            double *stB = new double[6];
            stB = ((double *) lammps_extract_compute(lmp,"stL1",0,1));         //Bottom surface stress
            double *stU = new double[6];
            stU = ((double *) lammps_extract_compute(lmp,"stL2",0,1));         //Top surface stress
            
        if (taskid==MASTER && Ktstep%10==0) outstress<<prev_Ktstep + Ktstep<<"\t"<<std::setprecision(10)<<KMCtime<<"\t"<<stB[0]/Vbulk<<"\t"<<stB[1]/Vbulk<<"\t"<<stB[2]/Vbulk<<"\t"<<stB[3]/Vbulk<<"\t"<<stB[4]/Vbulk<<"\t"<<stB[5]/Vbulk<<"\t"<<stU[0]/Vsurf<<"\t"<<stU[1]/Vsurf<<"\t"<<stU[2]/Vsurf<<"\t"<<stU[3]/Vsurf<<"\t"<<stU[4]/Vsurf<<"\t"<<stU[5]/Vsurf<<std::endl;
            
            lmp->input->one("fix              Freeze_1 Layer setforce 0.0 0.0 0.0");
            */
            /*########### STRESS ############*/
            
            lmp->input->one("create_atoms     3 region Box");
            lmp->input->one("pair_coeff       1 1 mie/cut 0 100 28 14 50");
            lmp->input->one("set              group all mass v_n100");
            lmp->input->one("set              group all diameter 100");
            lmp->input->one("set              group Nuc_big diameter 105");
            lmp->input->one("set              group Nuc_small diameter 95");
            lmp->input->one("pair_coeff       1 2 table 1-2.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       1 3 table 1-2.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       2 3 table 2-2.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       2 5 table 2-2.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       3 5 table 2-2.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       3 3 mie/cut 0 100 28 14 50");
            
            lmp->input->one("pair_coeff       1 6 table 1-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       2 6 table 2-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       3 6 table 2-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       5 6 table 2-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       6 6 table 6-6.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       6 7 table 2-2.em.dat EM28_14 185");
            
            lmp->input->one("pair_coeff       1 7 table 1-7.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       2 7 table 2-7.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       3 7 table 2-7.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       5 7 table 2-7.em.dat EM28_14 185");
            lmp->input->one("pair_coeff       7 7 table 7-7.em.dat EM28_14 185");
            lmp->input->one("run              0");
        }   /* End of Ktstep loop */
    
        delete lmp;
        end = clock();
    
    if (taskid == MASTER){
        outhistory <<"Execution time: "<< (double)(end-start)/CLOCKS_PER_SEC <<" seconds "<<(double)(end-start)*(1/3600.)/CLOCKS_PER_SEC<<" hours"<< std::endl;
        outhistory.close();
        outdump.close();
        outtest.close();
        outstress.close();
        printf("\nSUCCESS\n\n");
    }
    
    #ifdef PARALLEL
    MPI_Finalize();
    #endif

}   /* end of main */

// A FUNCTION TO READ THE attoms IN SIM, MOLECULES IN SOLUTION, AND POSSIBLE CHEMICAL REACTIONS 
void read_chem(std::vector<Atom2> &attoms,std::vector<MolSol> &molsol, std::vector<PartType> &parttype, std::vector<ChemRX> &chemRX){
    // this function should read a file containing info regarding the overall attoms that can be present, the composition of the solution, and particles that precipitate/dissolve.
    
	// For now, fast and dirty... I will specify only a cement-like solution with only one CSH and CH as possible particle..
	
	
    //  All attoms
    //  - number of Atom2 species in the whole simulation (e.g. 4 : Ca, Si, H, O in a C3S+H20 system)
    //  - name of each Atom2 species
    
	Atom2 temp_Atom2;
	temp_Atom2.name = "Ca";
	temp_Atom2.mass = 40.078 / AN * 1.E+24;  //in yg, i.e. 10^-24 grams 
	attoms.push_back(temp_Atom2);
	temp_Atom2.name.clear();
	temp_Atom2.name = "Si";
	temp_Atom2.mass = 28.08550 / AN * 1.E+24;  //in yg, i.e. 10^-24 grams 
	attoms.push_back(temp_Atom2);
	temp_Atom2.name.clear();
	temp_Atom2.name = "O";
	temp_Atom2.mass = 15.9994 / AN * 1.E+24;  //in yg, i.e. 10^-24 grams 
	attoms.push_back(temp_Atom2);
	temp_Atom2.name.clear();
	temp_Atom2.name = "H";
	temp_Atom2.mass = 1.00794 / AN * 1.E+24;  //in yg, i.e. 10^-24 grams 
	attoms.push_back(temp_Atom2);
	temp_Atom2.name.clear();
	
    //  Solution
    //  - number of molecular species that can be in solution (e.g. 4: Ca, H2SiO4 , H2O, OH)
    //  - name of each molecular species
    //  - composition of each molecular species in solution, i.e. number of attoms of each possible species composing the molecule
    //  - attributes of the molecules in solution: all what is needed to compute the free energy of the solution
    //  - initial concentration of each molecular species in solution (number per unit volume)
    
	MolSol temp_mol;
	temp_mol.name="Ca2+";
	temp_mol.atinm.push_back(1);
	temp_mol.atinm.push_back(0);
	temp_mol.atinm.push_back(0);
	temp_mol.atinm.push_back(0);
	temp_mol.conc = 20.E-3 * (1.E-24 * AN); //molecules per nm3... the value in moles times E-24 to convert per-liters to per-nm^3, and AN is the avogadro's number
	temp_mol.Ei = 0.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.Es = 100.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.q = 2.;
	molsol.push_back(temp_mol);
	temp_mol.name.clear();
	temp_mol.atinm.clear();
	
	temp_mol.name="H2SiO4";
	temp_mol.atinm.push_back(0);
	temp_mol.atinm.push_back(1);
	temp_mol.atinm.push_back(4);
	temp_mol.atinm.push_back(2);
	temp_mol.conc = 20.E-6 * (1.E-24 * AN);
	temp_mol.Ei = 600.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.Es = 100.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.q = -2.;
	molsol.push_back(temp_mol);
	temp_mol.name.clear();
	temp_mol.atinm.clear();
	
	temp_mol.name="H2O";
	temp_mol.atinm.push_back(0);
	temp_mol.atinm.push_back(0);
	temp_mol.atinm.push_back(1);
	temp_mol.atinm.push_back(2);
	temp_mol.conc = 55.56 * (1.E-24 * AN);
	temp_mol.Ei = 200.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.Es = 100.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.q = 0.;
	molsol.push_back(temp_mol);
	temp_mol.name.clear();
	temp_mol.atinm.clear();
	
	temp_mol.name="OH";
	temp_mol.atinm.push_back(0);
	temp_mol.atinm.push_back(0);
	temp_mol.atinm.push_back(1);
	temp_mol.atinm.push_back(1);
	temp_mol.conc = (40.E-3 -40.E-6) * (1.E-24 * AN);
	temp_mol.Ei = 100.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.Es = 100.;	// yg * nm^2/ps^2 =  10-27 Kg * 10 -18 m^2 / 10-24 s^2 = 10^-21 Joules = 0.00624150934 eV
	temp_mol.q = -1.;
	molsol.push_back(temp_mol);
	temp_mol.name.clear();
	temp_mol.atinm.clear();
	
    //  Particle types
    //  - number of possible particles types that can precipitate (e.g 3: CaOH2 , CSH(II), Substrate)
    //  - name of each particle type
    //  - attributes of each particle type:
    //      . number of attoms of each species per unit volume of particle
    //      . internal energy per unit volume (could be reassigned to each particle during simulation)
    //      . something about density of states and/or vibration freqs,
    //      . surface energy (in future this should depend on a solution chemistry that might change during the simulation. That could require in future a separate function to compute them..)
    //      . list of active chemical reactions for: (1) nucleation (2) growth (3) dissolution (4) leaching (5) sequestration (6) solid-solid mass transfer
    //  - number of possible nuclei for each particle type
    //  - size, shape, and orientation of possible nuclei for each particle type
	
	std::array<double, 3>	tarr;
	std::array<double, 6>	tarr2;
	
	PartType temp_ptype;
	temp_ptype.name = "CSH(II)";
	temp_ptype.ptid = 1;    //whatever you please. This is NOT the id that will be passed to LAMMPS.
	temp_ptype.atins.push_back(2);
	temp_ptype.atins.push_back(1);
	temp_ptype.atins.push_back(9);
	temp_ptype.atins.push_back(10);
	temp_ptype.volmol = (161.2 * 1.E+21 / AN);  //from Bullard, JAmCerSoc 2008, 161.2 cm3/mol
	temp_ptype.Eisp = 1.E+6; // 10^-21 J/nm3 = 0.00624150934 eV/nm^3. Invented value here
	temp_ptype.Essp = 1.E+4; // 10^-21 J/nm3 = 0.00624150934 eV/nm^2. Invented value here
	temp_ptype.dens = 2650.; // Kg m-3 = 10-24 g nm-3 = yg nm-3
	temp_ptype.RXnuc.push_back(0);	//the pushed-back id refers to the chemical reactions specified below
	temp_ptype.RXgrow.push_back(0);
	temp_ptype.RXdiss.push_back(1);
	// in this example, I will try nuclei with diameter 0.9 nm and 1.6 nm
	temp_ptype.nuclsize.push_back(0.9);
	//temp_ptype.nuclsize.push_back(1.6);
	// in this example I will allow two possible orientations of the nuclei, totally invented
	tarr[0]=0.1; tarr[1]=0.4; tarr[2]=0.87;		
	temp_ptype.nuclori.push_back( tarr );
	//tarr[0]=-0.5; tarr[1]=0.34; tarr[2]=0.7;
	//temp_ptype.nuclori.push_back( tarr );
	// in this example I will allow two possible shapes of the nuclei, totally invented
	temp_ptype.nuclshape.push_back("Sphere");
	tarr2[0]=1.; tarr2[1]=0.; tarr2[2]=0.; tarr2[3]=0.; tarr2[4]=0.; tarr2[5]=0.;	
	temp_ptype.shape_attr.push_back( tarr2 );  //this should be managed more smartly when reading an input file..
	temp_ptype.n_attr.push_back(1);
	//temp_ptype.nuclshape.push_back("Ellipsoid");
	//tarr2[0]=1.; tarr2[1]=2.; tarr2[2]=0.5; tarr2[3]=0.; tarr2[4]=0.; tarr2[5]=0.;
	//temp_ptype.shape_attr.push_back( tarr2 );
	//temp_ptype.n_attr.push_back(3);
	parttype.push_back(temp_ptype);
	temp_ptype.name.clear();
	temp_ptype.atins.clear();
	temp_ptype.RXnuc.clear();
	temp_ptype.RXgrow.clear();
	temp_ptype.RXdiss.clear();
	temp_ptype.nuclsize.clear();
	temp_ptype.nuclori.clear();
	temp_ptype.nuclshape.clear();
	temp_ptype.shape_attr.clear();
	temp_ptype.n_attr.clear();
	

    //  Chemical reactions
    //  invlving molecules in solutions and attoms in the solid (can be extended to include also solid-solid mass transfers)
    //  - number of possible chemical reactions
    //  - stoichiometric coefficients (this should say how many molecules are going in solution and how many attoms are produced in the solid particle)
    //  - activation energy of the reaction in isolated conditions
	//  - change of volume of solid induced by one reaction
    
	ChemRX temp_chemrx;
	temp_chemrx.name="CSH(II)prec"; //all from Bullard JAmCerSoc 2008
	temp_chemrx.mtosol.push_back(-2);
	temp_chemrx.mtosol.push_back(-1);
	temp_chemrx.mtosol.push_back(-3);
	temp_chemrx.mtosol.push_back(-2);
	temp_chemrx.atosld.push_back(2);
	temp_chemrx.atosld.push_back(1);
	temp_chemrx.atosld.push_back(9);
	temp_chemrx.atosld.push_back(10);
	temp_chemrx.DEiso = 1000. * (10.E+21 / AN);	//units of 0.00624150934 eV. The value in J/mol is converted to 10^-21 J by mutiplying times 10^21 and dividing by the avogadro's number 
	temp_chemrx.DVsld = 0.27;  //nm3... this is close to the molecular volume of CSH(II)
	chemRX.push_back(temp_chemrx);
	temp_chemrx.name.clear();
	temp_chemrx.mtosol.clear();
	temp_chemrx.atosld.clear();
	
}

// A FUNCTION TO READ THE INITIAL LAMMPS-LIKE CONFIGURATION
void read_conf(std::string c0fname , double box[], std::vector<Particle> &parts){
    // this function should read a lammps configuration including:
    //  - number of particles
    //  - number of particle types
    //  - box sizes
    //  - type of pairs to define subsequent coefficients
    //  - list of pair coefficients for each Atom2 type (WHAT IS THIS??).
    //  - type of attoms style (sphere..)
    //  - rows with label, type, diameter, density, x , y ,z , bs, bs ,bs
    
    // For now, I just write the conf here myself: fast and dirty..
    
    // box sizes (here in nanometers)
    box[0]=100;
    box[1]=100;
    box[2]=100;
    
    // number of attoms in the substrate layer
    int npart;
    double DAtom2 = 5.; //nm
    int npart_x = (int)(box[0]/DAtom2);
    int npart_y = (int)(box[1]/DAtom2);
    npart = npart_x * npart_y;
	
	//interparticle distances in the layer
	double dx = box[0]/(double)npart_x;
	double dy = box[1]/(double)npart_y;
    

	Particle temp_part;
	//create initial substrate particles
    for (int i=0 ; i<npart ; i++){
		temp_part.type = 4;	//"4" is the id that I assigned to the substrate particles beforehead
        temp_part.diam = 5.; //nm
		temp_part.dens = 2650.; // Kg m-3 = 10-24 g nm-3 = yg nm-3
		temp_part.mass = temp_part.dens * M_PI * temp_part.diam * temp_part.diam * temp_part.diam / 6.; //yg = 10-24 g
		temp_part.x = ((double)(i % npart_x) + 0.5) * dx;
		temp_part.y = ((double)( (int)(i/npart_x)) + 0.5) * dy;
		temp_part.z = 0.;
		temp_part.vx = 0.;
		temp_part.vy=0.;
		temp_part.vz=0;
		temp_part.o1=0.;
		temp_part.o2=0.;
		temp_part.o3=0.;
		temp_part.shape = "Sphere";
		temp_part.n_attr = 1;
		temp_part.sh[0]= 1.;
		temp_part.sh[1]= 0.; temp_part.sh[2]= 0.; temp_part.sh[3]= 0.; temp_part.sh[4]= 0.; temp_part.sh[5]= 0.;
		parts.push_back(temp_part);
    }
	//create 10 initial CSH(II) particles
	for (int i=0 ; i<10 ; i++){
		temp_part.type = 1;	//"1" is the id that I assigned to the CSH(II) particles beforehead
        temp_part.diam = 5.; //nm
		temp_part.dens = 2650.; // Kg m-3 = 10-24 g nm-3 = yg nm-3
		temp_part.mass = temp_part.dens * M_PI * temp_part.diam * temp_part.diam * temp_part.diam / 6.; //yg = 10-24 g
		temp_part.x = 0.01+box[0]/10.*sqrt((double)i);
		temp_part.y = 0.02+box[0]/15.*(double)i;
		temp_part.z = 20. + (double)(i%2)*30.;
		temp_part.vx = 0.;
		temp_part.vy=0.;
		temp_part.vz=0;
		temp_part.o1=0.;
		temp_part.o2=0.;
		temp_part.o3=0.;
		temp_part.shape = "Sphere";
		temp_part.n_attr = 1;
		temp_part.sh[0]= 1.;
		temp_part.sh[1]= 0.; temp_part.sh[2]= 0.; temp_part.sh[3]= 0.; temp_part.sh[4]= 0.; temp_part.sh[5]= 0.;
		parts.push_back(temp_part);
    }
	//create 2 initial CH particles
	for (int i=0 ; i<2 ; i++){
		temp_part.type = 8;	//"8" is the id that I assigned to the CH particles beforehead
        temp_part.diam = 7.; //nm
		temp_part.dens = 2260.; // Kg m-3 = 10-24 g nm-3 = yg nm-3
		temp_part.mass = temp_part.dens * M_PI * temp_part.diam * temp_part.diam * temp_part.diam / 6.; //yg = 10-24 g
		temp_part.x = 10.+ (double)i*20;
		temp_part.y = 10.+ (double)i*20;
		temp_part.z = 30. + (double)(i%2)*30.;
		temp_part.vx = 0.;
		temp_part.vy=0.;
		temp_part.vz=0;
		temp_part.o1=0.;
		temp_part.o2=0.;
		temp_part.o3=0.;
		temp_part.shape = "Sphere";
		temp_part.n_attr = 1;
		temp_part.sh[0]= 1.;
		temp_part.sh[1]= 0.; temp_part.sh[2]= 0.; temp_part.sh[3]= 0.; temp_part.sh[4]= 0.; temp_part.sh[5]= 0.;
		parts.push_back(temp_part);
    }
}


// A FUNCTION TO READ THE CHEMICAL INFO OF THE PARTICLES IN THE INITIAL CONFIGURATION (extra info to what lammps handles)
void read_Pchem(std::vector<Particle> &parts, std::vector<PartType> &parttype, std::vector<Atom2> &attoms){

	// for now, I just read the particle chemical info referring to the corresponding particle type defined previously
	// in future, for simulation restart purposes, it will be better to read also the chemical info of each particles from a file
	// saved along with the xyz coordinates. In fact, in future, the specific chemical composition of single particles might depart from
	// that in PartType as a consequence of leaching, enrichment, possibly growth and dissolution too.

	
	
	for (int i=0 ; i<parts.size() ; i++){
		
		//VOLUME AND SURFACE ARE COMPUTED ASSUMING SPHERICAL PARTICLES!!
		parts[i].vol = M_PI / 6. * parts[i].diam * parts[i].diam * parts[i].diam;
		parts[i].surf = M_PI * parts[i].diam * parts[i].diam;
		
		//number of attoms of different species
		for (int j=0; j<attoms.size(); j++ ) {
			parts[i].atinp.push_back( parts[i].vol/parttype[parts[i].typepos].volmol  * parttype[parts[i].typepos].atins[j] );
		}
		
		parts[i].Ei = parttype[parts[i].typepos].Eisp * parts[i].vol;
		parts[i].Esurf = parttype[parts[i].typepos].Essp * parts[i].surf;
	}
}

// A FUNCTION TO PRINT ALL THE INPUTS AND CHECK THAT ALL IS WORKING FINE
void print_input(std::vector<Atom2> &attoms,std::vector<MolSol> &molsol, std::vector<PartType> &parttype, std::vector<ChemRX> &chemRX, double box[], std::vector<Particle> &parts){
	
	
	printf("\n %i Atom2 species specified : ", (int)attoms.size());
	for (int i=0; i<attoms.size(); i++) {printf(" %s ", attoms[i].name.c_str());}
	printf("\n");
	printf("\n Their masses are : ");
	for (int i=0; i<attoms.size(); i++) {printf(" %f ", attoms[i].mass);}
	printf("\n");
	printf("\n %i species of molecules in solution : ", (int)molsol.size());
	for (int i=0; i<molsol.size(); i++) {printf(" %s ", molsol[i].name.c_str());}
	printf("\n");
	printf("\n Their chemical formulas are : ");
	for (int i=0; i<molsol.size(); i++) {
		for (int j=0;j<attoms.size(); j++) {
			if (molsol[i].atinm[j]>0) {
				printf("%1.1f%s",molsol[i].atinm[j],attoms[j].name.c_str());
			}
		}
		printf("  ");
	}
	printf("\n");
	printf("\n Their internal and solvation energies and charges are :");
	for (int i=0; i<molsol.size(); i++) {printf("\n %f %f %f", molsol[i].Ei,molsol[i].Es,molsol[i].q);}
	printf("\n");
	printf("\n %i possible chemical reactions : ", (int)chemRX.size());
	for (int i=0; i<chemRX.size(); i++) {printf(" %s ", chemRX[i].name.c_str());}
	printf("\n");
	printf("\n The reaction formulas are : \n");
	for (int i=0; i<chemRX.size(); i++) {
		for (int j=0;j<chemRX[i].mtosol.size(); j++) {
			if (chemRX[i].mtosol[j]!=0) {
				printf("%1.1f%s +",-chemRX[i].mtosol[j],molsol[j].name.c_str());
			}
		}
		printf(" --> ");
		for (int j=0;j<chemRX[i].atosld.size(); j++) {
			if (chemRX[i].atosld[j]!=0) {
				printf("%1.1f%s",chemRX[i].atosld[j],attoms[j].name.c_str());
			}
		}
		printf("\n");
	}
	printf("\n");
	printf("\n Their activation energies are :");
	for (int i=0; i<chemRX.size(); i++) {printf(" %f ", chemRX[i].DEiso);}
	printf("\n");
	printf("\n Their associated change in solid volume are :");
	for (int i=0; i<chemRX.size(); i++) {printf(" %f ", chemRX[i].DVsld);}
	printf("\n");
	printf("\n %i possible particle types : ", (int)parttype.size());
	for (int i=0; i<parttype.size(); i++) {printf(" %s ", parttype[i].name.c_str());}
	printf("\n");
	printf("\n Their chemical formulas are : ");
	for (int i=0; i<parttype.size(); i++) {
		bool any_form = false;
		for (int j=0;j<attoms.size(); j++) {
			if (parttype[i].atins[j]>0) {
				any_form=true;
				printf("%1.1f%s",parttype[i].atins[j],attoms[j].name.c_str());
			}
		}
		if (!any_form) {
			printf("none");
		}
		printf("  ");
	}
	printf("\n");
	printf("\n Their associated molecular volume and specific internal and surface energies :");
	for (int i=0; i<parttype.size(); i++) {printf("\n %f %f %f ", parttype[i].volmol, parttype[i].Eisp, parttype[i].Essp);}
	printf("\n");
	printf("\n Their possible nucleation reactions are :");
	for (int i=0; i<parttype.size(); i++) {
		printf("\n");
		for (int j=0; j<parttype[i].RXnuc.size(); j++) {
			printf("%s ", chemRX[ parttype[i].RXnuc[j] ].name.c_str());
		}
		if (parttype[i].RXnuc.size()==0) {
			printf("none");
		}
	}
	printf("\n");
	printf("\n Their possible growth reactions are :");
	for (int i=0; i<parttype.size(); i++) {
		printf("\n");
		for (int j=0; j<parttype[i].RXgrow.size(); j++) {
			printf("%s ", chemRX[ parttype[i].RXgrow[j] ].name.c_str());
		}
		if (parttype[i].RXgrow.size()==0) {
			printf("none");
		}
	}
	printf("\n");
	printf("\n Their possible dissolution reactions are :");
	for (int i=0; i<parttype.size(); i++) {
		printf("\n");
		for (int j=0; j<parttype[i].RXdiss.size(); j++) {
			printf("%s ", chemRX[ parttype[i].RXdiss[j] ].name.c_str());
		}
		if (parttype[i].RXdiss.size()==0) {
			printf("none");
		}
	}
	printf("\n");
	printf("\n Their possible nuclei sizes are :");
	for (int i=0; i<parttype.size(); i++) {
		printf("\n");
		for (int j=0; j<parttype[i].nuclsize.size(); j++) {
			printf("%f ", parttype[i].nuclsize[j]);
		}
		if (parttype[i].nuclsize.size()==0) {
			printf("none");
		}
	}
	printf("\n");
	printf("\n Their possible nuclei orientations are :");
	for (int i=0; i<parttype.size(); i++) {
		printf("\n");
		for (int j=0; j<parttype[i].nuclori.size(); j++) {
			printf("{ %f %f %f }  ", parttype[i].nuclori[j][0], parttype[i].nuclori[j][1], parttype[i].nuclori[j][2] );
		}
		if (parttype[i].nuclori.size()==0) {
			printf("none");
		}
	}
	printf("\n");
	printf("\n Their possible nuclei shapes are :");
	for (int i=0; i<parttype.size(); i++) {
		printf("\n");
		for (int j=0; j<parttype[i].nuclshape.size(); j++) {
			printf("%s : { ",parttype[i].nuclshape[j].c_str());
			for (int k=0; k<parttype[i].n_attr[j]; k++) {
				printf(" %f  ", parttype[i].shape_attr[j][k]);
			}
			printf("}    ");
		}
		if (parttype[i].nuclshape.size()==0) {
			printf("none");
		}
	}
	printf("\n");
	
	
	// all particles
	printf("\nBox sizes are %f %f %f \n",box[0],box[1],box[2]);
	printf("\n number of particles is %i \n",(int)parts.size());
	printf("\n ID  type   type_id      type_pos   ");
	for (int i=0; i<attoms.size(); i++) { printf("%s    ",attoms[i].name.c_str()); }
	printf("vol   surf    diam    mass    dens     x    y    z    vx     vy     vz     Ei    Esurf    o1     o2     o3    shape     n_attr   ");
	for (int i=0; i<MAX_n_shape_attr; i++) { printf("sh%i    ",i); }
	for (int i=0 ; i<parts.size() ; i++){
		printf("\n %i   %s   %i    %i    ",i,parttype[parts[i].typepos].name.c_str(),parts[i].type,parts[i].typepos);
		for (int j=0; j<attoms.size(); j++) { printf("%f    ",parts[i].atinp[j]); }
		printf(" %f   %f   %f    %f     %f   %f    %f     %f   ",parts[i].vol,parts[i].surf,parts[i].diam,parts[i].mass,parts[i].dens,parts[i].x,parts[i].y,parts[i].z);
		printf(" %f   %f   %f    %f     %f   %f    %f     %f   ",parts[i].vx,parts[i].vy,parts[i].vz,parts[i].Ei,parts[i].Esurf,parts[i].o1,parts[i].o2,parts[i].o3);
		printf(" %s   %i   ",parts[i].shape.c_str(),parts[i].n_attr);
		for (int j=0; j<MAX_n_shape_attr; j++) { printf("%f    ",parts[i].sh[j]); }
	}
	printf("\n");
}
