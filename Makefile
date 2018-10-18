
# Makefile generated automatically by /home/frederic/Work/Software/generator-framework/mkmk.pl '-l' '-lfastjetcontribfragile'
# run 'make make' to update it if you add new files

# get material from common/ directory
LCLINCLUDE   += -I/home/frederic/Work/Software/generator-framework
LIBRARIES += -L/home/frederic/Work/Software/generator-framework -lhepmcplus

include /home/frederic/Work/Software/generator-framework/Makefile.inc

LIBRARIES += -lfastjetcontribfragile

LIBRARIES += -lhepmcplus
INCLUDE += $(LCLINCLUDE)
INCLUDE += -DCOMMONDIR='"/home/frederic/Work/Software/generator-framework/"'

COMMONSRC = 
F77SRC = 
COMMONOBJ = $(patsubst %.cc,%.o,$(COMMONSRC)) $(patsubst %.f,%.o,$(F77SRC))

PROGSRC = cross-section-python.cc
PROGOBJ = $(patsubst %.cc,%.o,$(PROGSRC))


cross-section-python: cross-section-python.o /home/frederic/Work/Software/generator-framework/libhepmcplus.a $(COMMONOBJ)
	$(CXX) $(LDFLAGS) -o $@ $@.o $(COMMONOBJ) $(LIBRARIES) 


make:
	/home/frederic/Work/Software/generator-framework/mkmk.pl '-l' '-lfastjetcontribfragile'

clean:
	rm -vf $(COMMONOBJ) $(PROGOBJ)

.cc.o:         $<
	$(CXX) $(CXXFLAGS) -c $< -o $@
.f.o:         $<
	$(F77) $(FFLAGS) -c $< -o $@


depend:
	makedepend   $(LCLINCLUDE) -Y --   -- $(COMMONSRC) $(PROGSRC)
# DO NOT DELETE

cross-section-python.o: /home/frederic/Work/Software/generator-framework/AnalysisFramework.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/SimpleNTuple.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/SimpleHist.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/SimpleHist2D.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/AveragingHist.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/CorrelationHist.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/AverageAndError.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/NLOHistGeneric.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/CleverStream.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/CommonAnalysis.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/ExtractedEvent.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/addons.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/FlavourHolder.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/FJHepMC.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/FlexiBTagger.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/Calorimeter.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/DriverFromCmdLine.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/AnyDriver.hh
cross-section-python.o: /home/frederic/Work/Software/generator-framework/LockableMap.hh
cross-section-python.o: json.hpp cuts.hh
